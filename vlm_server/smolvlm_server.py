"""SmolVLM perception server: the robot's onboard VLM, answering closed visual questions over local HTTP.

Runs in the VLM venv (torch + transformers), separate from Isaac Sim's Python. Answers are scored from
the next-token probabilities (see smolvlm.py), so every answer is valid. --profile selects the onboard
computer it emulates (PROFILES below, or a JSON file given with --profiles): device, dtype, threads, CPU pinning, and the
latency the robot is charged. This endpoint will become a ROS 2 service on the robot.

    .venv/bin/python vlm_server/smolvlm_server.py --model models/SmolVLM-500M-Instruct --profile pi5_emulated
    POST /is_visible {"image_png_b64": "...", "phrase": "a red ball"} -> {"p_yes": 0.93, ...}
    POST /where      {"image_png_b64": "...", "phrase": "a red ball"} -> {"scores": {"left": .., "center": .., "right": ..}, ...}
    POST /is_near    {"image_png_b64": "...", "phrase": "a red ball"} -> {"p_yes": 0.12, ...}
    POST /ask        {"image_png_b64": "...", "entity": "table", "questions": [
                        {"id": "scene", "kind": "open", "text": "Describe this image in one sentence.", "max_tokens": 32},
                        {"id": "visible", "kind": "yesno", "text": "Is there a brown box in this image? ..."},
                        {"id": "action", "kind": "choice", "options": ["forward", "left", ...], "text": "...", "after": "scene"},
                        ...]}
                     -> {"answers": {"scene": {"answer": "A brown box ..."}, "visible": {"answer": "yes", "p_yes": 0.97},
                                     "action": {"answer": "forward", "scores": {...}}, ...}, ...}
                     several questions about one frame in one call (smolvlm.py ask / validate_questions)
    POST /feed       {"image_png_b64": "...", "frame_id": 12, "t": 6.0}  one frame of the robot's live camera
                     feed, shown in the live view; no model runs on it
    GET  /health     -> {"ok": true, "model": ..., "profile": ...}
    GET  /view       live view in a browser: each camera image, the question and the answer (live_view.py)
    POST /view/clear clear the live view's history
Every POST reply also has latency_s (measured here) and onboard_latency_s (latency_s x the profile's
latency_scale: what the robot is charged in simulation).
"""
import argparse
import base64
import io
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from smolvlm import QUESTION, SmolVLM, validate_questions  # noqa: E402,F401  (QUESTION re-exported for older imports)
from live_view import LiveLog, page  # noqa: E402

# Onboard-computer profiles: the compute the robot's VLM runs on. latency_scale multiplies the measured
# latency to give the latency the robot is charged in simulation (see docs/ONBOARD_VLM.md).
PROFILES = {
    "workstation_gpu": {
        # Development workstation GPU (RTX 4080). Not an onboard computer; the fast reference.
        # Same settings as the first prototype run (image splitting on).
        "device": "cuda", "dtype": "bfloat16", "threads": None, "cpu_affinity": None,
        "image_splitting": True, "latency_scale": 1.0,
    },
    "workstation_gpu_live": {
        # Workstation GPU for the live feed (run_prototype.py --approach live): image splitting off, so the live
        # question set (10 questions per frame) takes ~0.25 s with SmolVLM-500M-Instruct and ~0.5 s with
        # SmolVLM-Instruct (measured 2026-11-11): at least 1 analysed frame per second. Not an onboard computer.
        "device": "cuda", "dtype": "bfloat16", "threads": None, "cpu_affinity": None,
        "image_splitting": False, "latency_scale": 1.0,
    },
    "pi5_emulated": {
        # Raspberry Pi 5 stand-in: CPU only, fp32, 4 threads pinned to 4 i9-13900F efficiency cores (CPUs 16-19,
        # max 4.2 GHz). Pi 5: 4x Cortex-A76 at 2.4 GHz, no CUDA GPU. latency_scale is assumed (clock ratio
        # 4.2 / 2.4 GHz; the Pi's lower memory bandwidth is not modelled): replace with a measurement on the Pi.
        "device": "cpu", "dtype": "float32", "threads": 4, "cpu_affinity": [16, 17, 18, 19],
        "image_splitting": False, "latency_scale": 1.75,
    },
}


def load_profile(name, path=None):
    """A profile from PROFILES, or from a JSON file of {name: profile} when `path` is given."""
    profiles = PROFILES
    if path:
        with open(path, encoding="utf-8") as f:
            profiles = json.load(f)
    if name not in profiles or name.startswith("_"):
        raise SystemExit(f"unknown profile '{name}'; available: {[p for p in profiles if not p.startswith('_')]}")
    return profiles[name]


def apply_profile(profile):
    """Pin this process to the profile's CPUs (if any). Threads, device and dtype are applied by SmolVLM."""
    if profile.get("cpu_affinity"):
        os.sched_setaffinity(0, profile["cpu_affinity"])


def make_handler(vlm, profile_name, latency_scale):
    lock = Lock()  # one onboard computer: questions are answered one at a time
    live = LiveLog()
    def ask(img, entity, request):
        questions = request.get("questions")
        return {"answers": vlm.ask(img, questions), "questions": validate_questions(questions)}

    endpoints = {"/ask": ask,
                 "/is_visible": lambda img, p, r: {"p_yes": vlm.p_yes(img, p)},
                 "/is_near": lambda img, p, r: {"p_yes": vlm.p_near(img, p)},
                 "/where": lambda img, p, r: {"scores": vlm.where(img, p)}}

    class Handler(BaseHTTPRequestHandler):
        def _reply(self, code, payload, content_type="application/json"):
            body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            if self.path == "/health":
                self._reply(200, {"ok": True, "model": vlm.name, "profile": profile_name,
                                  "image_splitting": vlm.image_splitting, "quantize": vlm.quantize})
            elif self.path == "/view":
                self._reply(200, page().encode(), "text/html; charset=utf-8")
            elif self.path.startswith("/view/events"):
                after = int(self.path.partition("after=")[2] or 0)
                self._reply(200, {"model": vlm.name, "profile": profile_name, "events": live.events_after(after),
                                  "feed": live.feed_status(), "cleared_at": live.cleared_at})
            elif self.path.startswith("/view/feed.png"):
                frame = live.feed_frame()
                if frame is None:
                    self._reply(404, {"error": "no frame yet"})
                else:
                    self._reply(200, frame, "image/png")
            elif self.path.startswith("/view/image/") and self.path.endswith(".png"):
                image = live.image(int(self.path[len("/view/image/"):-4]))
                if image is None:
                    self._reply(404, {"error": "image no longer kept"})
                else:
                    self._reply(200, image, "image/png")
            else:
                self._reply(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802
            if self.path == "/view/clear":
                self._reply(200, {"ok": True, "last_id": live.clear()})
                return
            answer = endpoints.get(self.path)
            if answer is None and self.path != "/feed":
                self._reply(404, {"error": "not found"})
                return
            try:
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                png = base64.b64decode(request["image_png_b64"])
                image = Image.open(io.BytesIO(png)).convert("RGB")
                if self.path == "/feed":  # live view only: no model, no lock
                    live.feed(png, int(request.get("frame_id", 0)), float(request.get("t", 0.0)))
                    self._reply(200, {"ok": True})
                    return
                if self.path == "/ask":
                    phrase = str(request.get("entity", "")).strip()
                    validate_questions(request.get("questions"))
                else:
                    phrase = str(request["phrase"]).strip()
                    if not phrase:
                        raise ValueError("empty phrase")
            except Exception as error:  # malformed request
                self._reply(400, {"error": str(error)})
                return
            with lock:
                start = time.perf_counter()
                reply = answer(image, phrase, request)
                latency = time.perf_counter() - start
            reply.update(latency_s=latency, onboard_latency_s=latency * latency_scale, model=vlm.name,
                         profile=profile_name)
            live.record(self.path, phrase, image, reply)
            reply.pop("prompt", None)
            reply.pop("questions", None)
            self._reply(200, reply)

        def log_message(self, *args):
            pass

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/SmolVLM-Instruct")
    parser.add_argument("--profile", default="workstation_gpu", help="name in --profiles")
    parser.add_argument("--profiles", default=None, help="optional JSON file of profiles to use instead of PROFILES")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    profile = load_profile(args.profile, args.profiles)
    apply_profile(profile)
    vlm = SmolVLM(args.model, profile["device"], profile["dtype"], profile["image_splitting"], profile["threads"],
                  profile.get("quantize"))
    print(f"SmolVLM server: {vlm.name}, profile {args.profile} ({profile['device']}, {profile['dtype']}, "
          f"threads={profile['threads']}, cpus={profile['cpu_affinity']}, quantize={profile.get('quantize')}, "
          f"latency x{profile['latency_scale']}), "
          f"http://{args.host}:{args.port}  (live view: http://{args.host}:{args.port}/view)", flush=True)
    ThreadingHTTPServer((args.host, args.port), make_handler(vlm, args.profile, profile["latency_scale"])).serve_forever()


if __name__ == "__main__":
    main()
