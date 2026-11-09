"""SmolVLM perception server: the robot's onboard VLM, answering closed visual questions over local HTTP.

Runs in the VLM venv (torch + transformers), separate from Isaac Sim's Python. Answers are scored from
the next-token probabilities (see smolvlm.py), so every answer is valid. --profile selects the onboard
computer it emulates (config/onboard_profiles.json): device, dtype, threads, CPU pinning, and the
latency the robot is charged. This endpoint will become a ROS 2 service on the robot.

    .venv/bin/python vlm_server/smolvlm_server.py --model models/SmolVLM-256M-Instruct --profile pi5_emulated
    POST /is_visible {"image_png_b64": "...", "phrase": "a red ball"} -> {"p_yes": 0.93, ...}
    POST /where      {"image_png_b64": "...", "phrase": "a red ball"} -> {"scores": {"left": .., "center": .., "right": ..}, ...}
    POST /is_near    {"image_png_b64": "...", "phrase": "a red ball"} -> {"p_yes": 0.12, ...}
    GET  /health     -> {"ok": true, "model": ..., "profile": ...}
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
from smolvlm import QUESTION, SmolVLM  # noqa: E402,F401  (QUESTION re-exported for older imports)

DEFAULT_PROFILES = os.path.join(os.path.dirname(HERE), "config", "onboard_profiles.json")


def load_profile(name, path=DEFAULT_PROFILES):
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
    endpoints = {"/is_visible": lambda img, p: {"p_yes": vlm.p_yes(img, p)},
                 "/is_near": lambda img, p: {"p_yes": vlm.p_near(img, p)},
                 "/where": lambda img, p: {"scores": vlm.where(img, p)}}

    class Handler(BaseHTTPRequestHandler):
        def _reply(self, code, payload):
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            if self.path == "/health":
                self._reply(200, {"ok": True, "model": vlm.name, "profile": profile_name,
                                  "image_splitting": vlm.image_splitting, "quantize": vlm.quantize})
            else:
                self._reply(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802
            answer = endpoints.get(self.path)
            if answer is None:
                self._reply(404, {"error": "not found"})
                return
            try:
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                image = Image.open(io.BytesIO(base64.b64decode(request["image_png_b64"]))).convert("RGB")
                phrase = str(request["phrase"]).strip()
                if not phrase:
                    raise ValueError("empty phrase")
            except Exception as error:  # malformed request
                self._reply(400, {"error": str(error)})
                return
            with lock:
                start = time.perf_counter()
                reply = answer(image, phrase)
                latency = time.perf_counter() - start
            reply.update(latency_s=latency, onboard_latency_s=latency * latency_scale, model=vlm.name,
                         profile=profile_name)
            self._reply(200, reply)

        def log_message(self, *args):
            pass

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/SmolVLM-256M-Instruct")
    parser.add_argument("--profile", default="workstation_gpu", help="name in --profiles")
    parser.add_argument("--profiles", default=DEFAULT_PROFILES)
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
          f"http://{args.host}:{args.port}", flush=True)
    ThreadingHTTPServer((args.host, args.port), make_handler(vlm, args.profile, profile["latency_scale"])).serve_forever()


if __name__ == "__main__":
    main()
