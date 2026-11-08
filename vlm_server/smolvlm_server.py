"""SmolVLM perception server: answers "is <phrase> visible in this image?" over local HTTP.

Runs in the VLM venv (torch + transformers), separate from Isaac Sim's Python. Instead of letting
the model write free text, it compares the model's next-token probability of "Yes" against "No",
so every answer is valid and comes with a probability. This endpoint will become a ROS 2 service.

    .venv/bin/python vlm_server/smolvlm_server.py --model models/SmolVLM-Instruct
    POST /is_visible  {"image_png_b64": "...", "phrase": "a red apple"}
      -> {"p_yes": 0.93, "latency_s": 0.12, "model": "SmolVLM-Instruct"}
    GET  /health      -> {"ok": true, "model": ...}
"""
import argparse
import base64
import io
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

QUESTION = "Is there {phrase} in this image? Answer with yes or no."


class SmolVLM:
    def __init__(self, model_dir, device="cuda"):
        self.name = os.path.basename(os.path.normpath(model_dir))
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_dir)
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self.model = AutoModelForImageTextToText.from_pretrained(model_dir, dtype=dtype).to(device).eval()
        tokenizer = self.processor.tokenizer
        # First token of each spelling; the answer may start with or without a leading space.
        self.yes_ids = sorted({tokenizer.encode(w, add_special_tokens=False)[0] for w in ("Yes", " Yes", "yes", " yes")})
        self.no_ids = sorted({tokenizer.encode(w, add_special_tokens=False)[0] for w in ("No", " No", "no", " no")})
        self.lock = Lock()

    @torch.inference_mode()
    def p_yes(self, image, phrase):
        messages = [{"role": "user", "content": [{"type": "image"},
                                                 {"type": "text", "text": QUESTION.format(phrase=phrase)}]}]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=[image], return_tensors="pt").to(self.device)
        with self.lock:
            logits = self.model(**inputs).logits[0, -1].float()
        yes = torch.logsumexp(logits[self.yes_ids], 0)
        no = torch.logsumexp(logits[self.no_ids], 0)
        return float(torch.sigmoid(yes - no))


def make_handler(vlm):
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
                self._reply(200, {"ok": True, "model": vlm.name})
            else:
                self._reply(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802
            if self.path != "/is_visible":
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
            start = time.perf_counter()
            p = vlm.p_yes(image, phrase)
            self._reply(200, {"p_yes": p, "latency_s": time.perf_counter() - start, "model": vlm.name})

        def log_message(self, *args):
            pass

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/SmolVLM-Instruct")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    vlm = SmolVLM(args.model, args.device)
    print(f"SmolVLM server: {vlm.name} on {args.device}, http://{args.host}:{args.port}", flush=True)
    ThreadingHTTPServer((args.host, args.port), make_handler(vlm)).serve_forever()


if __name__ == "__main__":
    main()
