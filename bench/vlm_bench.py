"""Benchmark SmolVLM variants for onboard use: accuracy on labelled sim images, and speed/memory on an
emulated onboard computer.

    # accuracy (any device; fp32 by default so the arithmetic matches a CPU-only board)
    .venv/bin/python bench/vlm_bench.py accuracy --data bench/data/one_room_eval \
        --models models/SmolVLM-256M-Instruct models/SmolVLM-500M-Instruct --out bench/results
    # speed and peak memory under a profile (each model in its own process, pinned to the profile's CPUs)
    .venv/bin/python bench/vlm_bench.py latency --profile pi5_emulated --data bench/data/one_room_eval \
        --models models/SmolVLM-256M-Instruct --out bench/results
    # summary table from both
    .venv/bin/python bench/vlm_bench.py report --out bench/results

Tasks (questions in vlm_server/smolvlm.py):
  presence   "Is there <phrase> in this image?" for every object, visible or not
  where      left/center/right from three crops (where) and from one worded question (where_words)
  near       "Is <phrase> close to the camera?" for visible objects
Ground truth comes from Isaac Sim's tight 2D boxes (sim/capture_dataset.py): an object is visible if its
visible box is at least MIN_AREA px; smaller slivers are left out of the presence task as ambiguous.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "vlm_server"))

MIN_AREA = 150  # px; about 12 x 12 in a 320 x 240 frame
NEAR_HEIGHT = 0.25  # visible box at least this fraction of the image height counts as near
EDGE_MARGIN = 0.04  # objects whose centre is this close to a third boundary are left out of `where`
DIRECTIONS = ("left", "center", "right")
# visible-box area bins (px of a 320 x 240 frame) for recall by apparent size
AREA_BINS = (("small_150-1k", 150, 1000), ("medium_1k-5k", 1000, 5000), ("large_5k+", 5000, 10**9))


# -- ground truth ---------------------------------------------------------------------------------
def load_dataset(data_dir):
    with open(os.path.join(data_dir, "labels.jsonl"), encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def make_cases(frames):
    """Questions with expected answers, derived from the boxes. Labels never go to the model."""
    cases = []
    for fr in frames:
        w, h = fr["width"], fr["height"]
        for entity, phrase in fr["descriptions"].items():
            box = fr["boxes"].get(entity)
            area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1) if box else 0
            base = {"image": fr["image"], "entity": entity, "phrase": phrase, "area": area}
            if 0 < area < MIN_AREA:
                continue  # sliver: neither clearly visible nor clearly absent
            visible = area >= MIN_AREA
            cases.append({**base, "task": "presence", "expected": visible,
                          "decoy_in_view": any(e.startswith("decoy") for e in fr["boxes"])})
            if not visible:
                continue
            cx = (box[0] + box[2]) / 2 / w
            if min(abs(cx - 1 / 3), abs(cx - 2 / 3)) >= EDGE_MARGIN:
                direction = "left" if cx < 1 / 3 else "right" if cx > 2 / 3 else "center"
                cases.append({**base, "task": "where", "expected": direction, "cx": round(cx, 3)})
            cases.append({**base, "task": "near", "expected": (box[3] - box[1] + 1) / h >= NEAR_HEIGHT})
    return cases


# -- accuracy ---------------------------------------------------------------------------------------
def run_accuracy(args):
    from smolvlm import SmolVLM, load_image

    frames = load_dataset(args.data)
    cases = make_cases(frames)
    os.makedirs(args.out, exist_ok=True)
    for model_dir in args.models:
        for splitting in args.splitting:
            tag = f"{os.path.basename(os.path.normpath(model_dir))}{'-int8' if args.quantize else ''}__split{int(splitting)}"
            vlm = SmolVLM(model_dir, args.device, args.dtype, image_splitting=splitting, quantize=args.quantize)
            images = {}
            records = []
            t0 = time.perf_counter()
            for case in cases:
                image = images.setdefault(case["image"], load_image(os.path.join(args.data, "images", case["image"])))
                if case["task"] == "presence":
                    out = {"p_yes": vlm.p_yes(image, case["phrase"])}
                elif case["task"] == "near":
                    out = {"p_yes": vlm.p_near(image, case["phrase"]), "p_lowres": vlm.p_near_lowres(image, case["phrase"])}
                else:
                    out = {"crops": vlm.where(image, case["phrase"]), "words": vlm.where_words(image, case["phrase"])}
                records.append({**case, **out})
            with open(os.path.join(args.out, f"accuracy__{tag}.jsonl"), "w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")
            print(f"{tag}: {len(records)} questions in {time.perf_counter() - t0:.0f} s", flush=True)
            print(json.dumps(score(records), indent=1), flush=True)
            del vlm
            if args.device == "cuda":
                import torch
                torch.cuda.empty_cache()


def run_add_lowres(args):
    """Add p_lowres to the near cases of existing accuracy files (for files written before it existed)."""
    from smolvlm import SmolVLM, load_image

    for model_dir in args.models:
        for splitting in args.splitting:
            tag = f"{os.path.basename(os.path.normpath(model_dir))}__split{int(splitting)}"
            path = os.path.join(args.out, f"accuracy__{tag}.jsonl")
            with open(path, encoding="utf-8") as f:
                records = [json.loads(line) for line in f]
            if all("p_lowres" in r for r in records if r["task"] == "near"):
                continue
            vlm = SmolVLM(model_dir, args.device, args.dtype, image_splitting=splitting)
            for r in records:
                if r["task"] == "near":
                    r["p_lowres"] = vlm.p_near_lowres(load_image(os.path.join(args.data, "images", r["image"])), r["phrase"])
            with open(path, "w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")
            print(f"{tag}: near_lowres {json.dumps(score(records)['near_lowres'])}", flush=True)


def _balanced(records, threshold=0.5):
    tp = sum(r["p_yes"] >= threshold and r["expected"] for r in records)
    fn = sum(r["p_yes"] < threshold and r["expected"] for r in records)
    tn = sum(r["p_yes"] < threshold and not r["expected"] for r in records)
    fp = sum(r["p_yes"] >= threshold and not r["expected"] for r in records)
    recall = tp / (tp + fn) if tp + fn else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    return {"n": len(records), "positives": tp + fn, "recall": round(recall, 3), "specificity": round(specificity, 3),
            "balanced_accuracy": round((recall + specificity) / 2, 3), "false_yes": fp, "false_no": fn,
            "auc": _auc(records)}


def _auc(records):
    """ROC AUC of p_yes: chance that a random visible case scores above a random absent one (threshold-free)."""
    pos = [r["p_yes"] for r in records if r["expected"]]
    neg = [r["p_yes"] for r in records if not r["expected"]]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return round(wins / (len(pos) * len(neg)), 3)


def calibrated(records):
    """Pick the p_yes threshold with the best balanced accuracy on even-numbered images, report it on odd ones."""
    def image_index(r):
        return int(r["image"].split(".")[0])
    calib = [r for r in records if image_index(r) % 2 == 0]
    test = [r for r in records if image_index(r) % 2 == 1]
    candidates = sorted({round(r["p_yes"], 4) for r in calib} | {0.5})
    best = max(candidates, key=lambda t: _balanced(calib, t)["balanced_accuracy"])
    held_out = _balanced(test, best)
    return {"threshold": best, "held_out_balanced_accuracy": held_out["balanced_accuracy"],
            "held_out_recall": held_out["recall"], "held_out_specificity": held_out["specificity"]}


def score(records):
    presence = [r for r in records if r["task"] == "presence"]
    where = [r for r in records if r["task"] == "where"]
    near = [r for r in records if r["task"] == "near"]
    out = {"presence": _balanced(presence),
           "presence_calibrated": calibrated(presence),
           # look-alikes: a real target is asked about while it is absent but a decoy is in view
           "presence_lookalike": _balanced([r for r in presence if not r["entity"].startswith("decoy")
                                            and not r["expected"] and r.get("decoy_in_view")]),
           "presence_by_entity": {e: _balanced([r for r in presence if r["entity"] == e])
                                  for e in sorted({r["entity"] for r in presence})},
           "presence_recall_by_area": {
               label: _balanced([r for r in presence if r["expected"] and lo <= r["area"] < hi])["recall"]
               for label, lo, hi in AREA_BINS},
           "near": _balanced(near),
           "near_calibrated": calibrated(near) if near else {},
           "near_lowres": _balanced([{**r, "p_yes": r["p_lowres"]} for r in near]) if near and "p_lowres" in near[0] else {}}
    if where:
        crops_ok = sum(max(r["crops"], key=r["crops"].get) == r["expected"] for r in where)
        crops_seen = sum(max(r["crops"].values()) >= 0.5 for r in where)
        words_ok = sum(max(r["words"], key=r["words"].get) == r["expected"] for r in where)
        out["where"] = {"n": len(where), "crops_accuracy": round(crops_ok / len(where), 3),
                        "crops_target_seen_in_some_crop": round(crops_seen / len(where), 3),
                        "words_accuracy": round(words_ok / len(where), 3),
                        "chance": round(max(sum(r["expected"] == d for r in where) for d in DIRECTIONS) / len(where), 3)}
    return out


# -- latency and memory -------------------------------------------------------------------------------
def run_latency(args):
    """Launch one child per (model, splitting) so pinning and peak memory are measured per model."""
    from smolvlm_server import load_profile

    profile = load_profile(args.profile)
    os.makedirs(args.out, exist_ok=True)
    results = []
    for model_dir in args.models:
        for splitting in args.splitting:
            cmd = [sys.executable, __file__, "_latency_child", "--profile", args.profile, "--data", args.data,
                   "--model", model_dir, "--n", str(args.n), "--split", str(int(splitting))]
            if args.quantize:
                cmd += ["--quantize", args.quantize]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            line = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
            if proc.returncode or not line:
                print(f"{model_dir} split={splitting}: FAILED\n{proc.stderr[-2000:]}", flush=True)
                continue
            result = json.loads(line[-1])
            result["latency_scale"] = profile["latency_scale"]
            results.append(result)
            print(json.dumps(result), flush=True)
    path = os.path.join(args.out, f"latency__{args.profile}.json")
    merged = {}
    if os.path.exists(path):  # keep earlier runs; a re-measured (model, splitting) replaces its old entry
        with open(path, encoding="utf-8") as f:
            merged = {(r["model"], r["image_splitting"]): r for r in json.load(f)["results"]}
    merged.update({(r["model"], r["image_splitting"]): r for r in results})
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"profile": args.profile, "profile_spec": profile, "results": list(merged.values())}, f, indent=2)


def _latency_child(args):
    import resource

    from smolvlm import SmolVLM, load_image
    from smolvlm_server import apply_profile, load_profile

    profile = load_profile(args.profile)
    apply_profile(profile)
    t0 = time.perf_counter()
    vlm = SmolVLM(args.model, profile["device"], profile["dtype"], bool(args.split), profile["threads"],
                  args.quantize or profile.get("quantize"))
    load_s = time.perf_counter() - t0
    frames = load_dataset(args.data)
    cases = [c for c in make_cases(frames) if c["task"] == "where"][: args.n + 1]
    timings = {"p_yes": [], "where_crops": [], "near": []}
    for i, case in enumerate(cases):
        image = load_image(os.path.join(args.data, "images", case["image"]))
        for key, fn in (("p_yes", vlm.p_yes), ("where_crops", vlm.where), ("near", vlm.p_near)):
            t = time.perf_counter()
            fn(image, case["phrase"])
            if i > 0:  # first round is warm-up
                timings[key].append(time.perf_counter() - t)
    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
    print(json.dumps({"model": vlm.name + ("-int8" if vlm.quantize else ""), "image_splitting": bool(args.split), "device": profile["device"],
                      "dtype": profile["dtype"], "threads": profile["threads"], "cpus": profile["cpu_affinity"],
                      "load_s": round(load_s, 2), "peak_rss_gb": round(peak_gb, 2), "n": len(timings["p_yes"]),
                      **{f"{k}_median_s": round(statistics.median(v), 3) for k, v in timings.items()},
                      **{f"{k}_max_s": round(max(v), 3) for k, v in timings.items()}}))


# -- report -------------------------------------------------------------------------------------------
def run_report(args):
    rows, records_by_tag = [], {}
    for name in sorted(os.listdir(args.out)):
        if name.startswith("accuracy__") and name.endswith(".jsonl"):
            with open(os.path.join(args.out, name), encoding="utf-8") as f:
                records = [json.loads(line) for line in f]
            model, split = name[len("accuracy__"):-len(".jsonl")].split("__split")
            rows.append((model, split == "1", score(records)))
            records_by_tag[(model, split == "1")] = records
    thresholds = {f"{model}__split{int(split)}": {"visible": s["presence_calibrated"]["threshold"],
                                                  "near": s["near_calibrated"]["threshold"]}
                  for model, split, s in rows}
    with open(os.path.join(args.out, "thresholds.json"), "w", encoding="utf-8") as f:
        json.dump({"_comment": "p_yes thresholds per model and image-splitting setting, chosen on even-numbered images "
                               "of the benchmark set (bench/vlm_bench.py report). Read by run_prototype.py --vlm-thresholds.",
                   **thresholds}, f, indent=2)
    latency = {}
    for name in os.listdir(args.out):
        if name.startswith("latency__"):
            with open(os.path.join(args.out, name), encoding="utf-8") as f:
                data = json.load(f)
            for r in data["results"]:
                latency[(r["model"], r["image_splitting"], data["profile"])] = r
    print("| Model | Image splitting | Presence AUC | Presence bal. acc. @0.5 | Recall | Recall small / medium / large | "
          "Specificity @0.5 | Look-alike false yes (calibrated) | Calibrated threshold -> held-out bal. acc. (recall / spec.) | "
          "Where (crops) | Where (words) | Near AUC | Near calibrated held-out bal. acc. |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for model, split, s in rows:
        p, d, w, n = s["presence"], s["presence_lookalike"], s.get("where", {}), s["near"]
        cal = s["presence_calibrated"]
        decoy_neg = d["n"]
        lookalike_cal = sum(r["p_yes"] >= cal["threshold"] for r in records_by_tag[(model, split)]
                            if r["task"] == "presence" and not r["entity"].startswith("decoy")
                            and not r["expected"] and r.get("decoy_in_view"))
        by_area = " / ".join(str(v) for v in s["presence_recall_by_area"].values())
        print(f"| {model} | {'on' if split else 'off'} | {p['auc']} | {p['balanced_accuracy']} | {p['recall']} | {by_area} | "
              f"{p['specificity']} | {lookalike_cal}/{decoy_neg} | {cal['threshold']:.3g} -> {cal['held_out_balanced_accuracy']} "
              f"({cal['held_out_recall']} / {cal['held_out_specificity']}) | {w.get('crops_accuracy')} (chance {w.get('chance')}) | "
              f"{w.get('words_accuracy')} | {n['auc']} | {s['near_calibrated'].get('held_out_balanced_accuracy')} |")
    if latency:
        print("\n| Model | Image splitting | Profile | Load s | Peak RAM GB | p_yes median s | where (3 crops) median s | "
              "Charged p_yes s (x scale) |")
        print("|---|---|---|---|---|---|---|---|")
        for (model, split, profile), r in sorted(latency.items()):
            print(f"| {model} | {'on' if split else 'off'} | {profile} | {r['load_s']} | {r['peak_rss_gb']} | "
                  f"{r['p_yes_median_s']} | {r['where_crops_median_s']} | "
                  f"{round(r['p_yes_median_s'] * r['latency_scale'], 2)} |")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["accuracy", "latency", "report", "add_lowres", "_latency_child"])
    parser.add_argument("--data", default=os.path.join(HERE, "data", "one_room_eval"))
    parser.add_argument("--models", nargs="+", default=[])
    parser.add_argument("--model")
    parser.add_argument("--out", default=os.path.join(HERE, "results"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--splitting", nargs="+", type=int, default=[0, 1], help="image splitting settings to test")
    parser.add_argument("--profile", default="pi5_emulated")
    parser.add_argument("--n", type=int, default=8, help="timed rounds per model (after one warm-up)")
    parser.add_argument("--split", type=int, default=0)
    parser.add_argument("--quantize", choices=["int8"], default=None, help="dynamic int8 weights (CPU only)")
    args = parser.parse_args()
    args.splitting = [bool(s) for s in args.splitting]
    {"accuracy": run_accuracy, "latency": run_latency, "report": run_report, "add_lowres": run_add_lowres,
     "_latency_child": _latency_child}[args.mode](args)


if __name__ == "__main__":
    main()
