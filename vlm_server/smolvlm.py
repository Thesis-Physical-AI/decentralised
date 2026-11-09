"""SmolVLM wrapper: closed visual questions answered by scoring the next token, never by free text.

Shared by vlm_server/smolvlm_server.py (the robot's perception service) and bench/vlm_bench.py.
Every method returns probabilities, so every answer is valid:
  p_yes(image, phrase)        P("yes") to "Is there <phrase> in this image?"
  where(image, phrase)        {"left", "center", "right"} -> P(yes) on three overlapping vertical crops
  where_words(image, phrase)  same labels, scored from one question that names the three options
  p_near(image, phrase)       P("yes") to "Is <phrase> close to the camera?"
  p_near_lowres(image, phrase) P("yes") to the presence question on a heavily downscaled image: far
                              objects shrink to a few pixels and drop out, near ones stay recognisable
"""
import os

import torch
from PIL import Image

QUESTION = "Is there {phrase} in this image? Answer with yes or no."
NEAR_QUESTION = "Is {phrase} close to the camera, taking up a large part of the image? Answer with yes or no."
WHERE_QUESTION = "Where is {phrase} in this image? Answer with one word: left, center, or right."
DIRECTIONS = ("left", "center", "right")
# Horizontal crops (fractions of the image width) used by where(); they overlap so an object on a
# boundary is fully inside at least one crop.
CROPS = {"left": (0.0, 0.45), "center": (0.275, 0.725), "right": (0.55, 1.0)}
LOWRES_FACTOR = 8  # p_near_lowres: 320 x 240 -> 40 x 30
DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def first_token_ids(tokenizer, words):
    """Token ids that start each spelling of each word (with/without a leading space, any case)."""
    ids = set()
    for word in words:
        for spelling in (word, " " + word, word.capitalize(), " " + word.capitalize()):
            ids.add(tokenizer.encode(spelling, add_special_tokens=False)[0])
    return sorted(ids)


class SmolVLM:
    def __init__(self, model_dir, device="cuda", dtype="bfloat16", image_splitting=False, threads=None, quantize=None):
        """quantize="int8": dynamic int8 weights for every Linear layer (CPU only), with the qnnpack engine
        that PyTorch uses on ARM boards such as the Raspberry Pi."""
        from transformers import AutoModelForImageTextToText, AutoProcessor

        if threads:
            torch.set_num_threads(threads)
        self.name = os.path.basename(os.path.normpath(model_dir))
        self.device = device
        self.image_splitting = image_splitting
        self.processor = AutoProcessor.from_pretrained(model_dir)
        self.processor.image_processor.do_image_splitting = image_splitting
        self.model = AutoModelForImageTextToText.from_pretrained(model_dir, dtype=DTYPES[dtype]).to(device).eval()
        self.quantize = quantize
        if quantize == "int8":
            if device != "cpu":
                raise ValueError("int8 dynamic quantization runs on the CPU only")
            torch.backends.quantized.engine = "qnnpack"
            self.model = torch.ao.quantization.quantize_dynamic(self.model, {torch.nn.Linear}, dtype=torch.qint8)
        elif quantize:
            raise ValueError(f"unknown quantize option {quantize!r}")
        tok = self.processor.tokenizer
        self.yes_ids, self.no_ids = first_token_ids(tok, ["yes"]), first_token_ids(tok, ["no"])
        self.direction_ids = {d: first_token_ids(tok, [d] + (["middle"] if d == "center" else [])) for d in DIRECTIONS}

    @torch.inference_mode()
    def _last_logits(self, image, text):
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=[image], return_tensors="pt").to(self.device)
        return self.model(**inputs).logits[0, -1].float()

    def _p_yes_text(self, image, text):
        logits = self._last_logits(image, text)
        return float(torch.sigmoid(torch.logsumexp(logits[self.yes_ids], 0) - torch.logsumexp(logits[self.no_ids], 0)))

    def p_yes(self, image, phrase):
        return self._p_yes_text(image, QUESTION.format(phrase=phrase))

    def p_near(self, image, phrase):
        return self._p_yes_text(image, NEAR_QUESTION.format(phrase=phrase))

    def p_near_lowres(self, image, phrase, factor=LOWRES_FACTOR):
        width, height = image.size
        small = image.resize((max(1, width // factor), max(1, height // factor)), Image.BILINEAR)
        return self.p_yes(small.resize((width, height), Image.NEAREST), phrase)

    def where(self, image, phrase):
        width, height = image.size
        scores = {}
        for direction, (x0, x1) in CROPS.items():
            crop = image.crop((round(x0 * width), 0, round(x1 * width), height))
            scores[direction] = self.p_yes(crop, phrase)
        return scores

    def where_words(self, image, phrase):
        logits = self._last_logits(image, WHERE_QUESTION.format(phrase=phrase))
        scores = torch.stack([torch.logsumexp(logits[self.direction_ids[d]], 0) for d in DIRECTIONS]).softmax(0)
        return dict(zip(DIRECTIONS, (float(s) for s in scores)))


def load_image(path):
    return Image.open(path).convert("RGB")
