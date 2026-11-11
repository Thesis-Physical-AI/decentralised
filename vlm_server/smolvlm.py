"""SmolVLM wrapper: visual questions about the robot's camera image.

Used by vlm_server/smolvlm_server.py (the robot's perception service).
Closed questions are scored from the next token, so every closed answer is valid and has a probability:
  p_yes(image, phrase)        P("yes") to "Is there <phrase> in this image?"
  where(image, phrase)        {"left", "center", "right"} -> P(yes) on three overlapping vertical crops
  p_near(image, phrase)       P("yes") to "Is <phrase> close to the camera?"
  ask(image, questions)       several questions about one image in one call (at most two batched passes):
                              open questions answered in generated text, yes/no and multiple-choice
                              questions scored from the next token. Used by the live feed.
"""
import os
import re

import torch
from PIL import Image

QUESTION = "Is there {phrase} in this image? Answer with yes or no."
NEAR_QUESTION = "Is {phrase} close to the camera, taking up a large part of the image? Answer with yes or no."
THOUGHT_PREFIX = "Thought:"  # ask(): a closed question with "after" sees that open answer as a thought
# Horizontal crops (fractions of the image width) used by where(); they overlap so an object on a
# boundary is fully inside at least one crop.
CROPS = {"left": (0.0, 0.45), "center": (0.275, 0.725), "right": (0.55, 1.0)}
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

    def _prompt(self, text):
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        return self.processor.apply_chat_template(messages, add_generation_prompt=True)

    @torch.inference_mode()
    def _last_logits(self, image, text):
        prompt = self._prompt(text)
        inputs = self.processor(text=prompt, images=[image], return_tensors="pt").to(self.device)
        return self.model(**inputs).logits[0, -1].float()

    @torch.inference_mode()
    def ask(self, image, questions):
        """Answer several questions about one image at once: {id: answer}. See validate_questions() for the
        question format. At most two batched passes: one generate() for all open questions, then one
        forward pass that scores every closed question (yesno, choice) from its next token. A closed
        question with "after": <open id> sees that open answer first, as "Thought: <answer>"."""
        questions = validate_questions(questions)
        answers = {}
        open_qs = [q for q in questions if q["kind"] == "open"]
        if open_qs:
            texts = self._batch(image, open_qs, [" " for _ in open_qs], generate=max(q["max_tokens"] for q in open_qs))
            for q, text in zip(open_qs, texts):
                answers[q["id"]] = {"answer": clean_answer(text)}
        closed = [q for q in questions if q["kind"] != "open"]
        if closed:
            prefixes = [f" {THOUGHT_PREFIX} {answers[q['after']]['answer']}\nAnswer:" if q.get("after") else ""
                        for q in closed]
            logits = self._batch(image, closed, prefixes)
            for q, row in zip(closed, logits):
                if q["kind"] == "yesno":
                    p = float(torch.sigmoid(torch.logsumexp(row[self.yes_ids], 0) - torch.logsumexp(row[self.no_ids], 0)))
                    answers[q["id"]] = {"answer": "yes" if p >= 0.5 else "no", "p_yes": p}
                else:
                    ids = self._option_ids(tuple(q["options"]))
                    probs = torch.stack([torch.logsumexp(row[ids[o]], 0) for o in q["options"]]).softmax(0)
                    scores = dict(zip(q["options"], (float(p) for p in probs)))
                    answers[q["id"]] = {"answer": max(scores, key=scores.get), "scores": scores}
        return answers

    def _batch(self, image, questions, prefixes, generate=0):
        """One batched pass over the questions (each with its own crop of the image, if any).
        generate=N: greedy-decode up to N tokens per row and return the texts; else last-token logits."""
        width, height = image.size
        images = [[image.crop((round(q["crop"][0] * width), 0, round(q["crop"][1] * width), height))
                   if q.get("crop") else image] for q in questions]
        texts = [self._prompt(q["text"]) + prefix for q, prefix in zip(questions, prefixes)]
        tokenizer = self.processor.tokenizer
        side, tokenizer.padding_side = tokenizer.padding_side, "left"  # last position = last real token
        try:
            inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True).to(self.device)
        finally:
            tokenizer.padding_side = side
        if generate:
            out = self.model.generate(**inputs, max_new_tokens=generate, do_sample=False,
                                      pad_token_id=tokenizer.pad_token_id)
            return tokenizer.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return self.model(**inputs).logits[:, -1].float()

    def _option_ids(self, options):
        cache = self.__dict__.setdefault("_option_cache", {})
        if options not in cache:
            cache[options] = distinct_first_token_ids(self.processor.tokenizer, options)
        return cache[options]

    def _p_yes_text(self, image, text):
        logits = self._last_logits(image, text)
        return float(torch.sigmoid(torch.logsumexp(logits[self.yes_ids], 0) - torch.logsumexp(logits[self.no_ids], 0)))

    def p_yes(self, image, phrase):
        return self._p_yes_text(image, QUESTION.format(phrase=phrase))

    def p_near(self, image, phrase):
        return self._p_yes_text(image, NEAR_QUESTION.format(phrase=phrase))

    def where(self, image, phrase):
        width, height = image.size
        scores = {}
        for direction, (x0, x1) in CROPS.items():
            crop = image.crop((round(x0 * width), 0, round(x1 * width), height))
            scores[direction] = self.p_yes(crop, phrase)
        return scores



def distinct_first_token_ids(tokenizer, words):
    """first_token_ids per word, without ids shared between words, so each score belongs to one option."""
    all_ids = {word: set(first_token_ids(tokenizer, [word])) for word in words}
    distinct = {}
    for word in words:
        shared = set().union(*(all_ids[other] for other in words if other != word))
        distinct[word] = sorted(all_ids[word] - shared)
        if not distinct[word]:
            raise ValueError(f"option '{word}' has no first token of its own in this tokenizer")
    return distinct


MAX_QUESTIONS, MAX_OPEN_TOKENS = 16, 64
KINDS = ("yesno", "choice", "open")


def validate_questions(questions):
    """Check a list of questions for ask(); returns normalised copies or raises ValueError.

    Each question: {"id": str, "kind": "yesno" | "choice" | "open", "text": str, plus
      "options": [str, ...]   choice: 2-12 one-word options, scored; the answer is always one of them
      "max_tokens": int       open: length cap of the generated answer (default 32, at most 64)
      "crop": [x0, x1]        optional: ask about this horizontal slice of the image (fractions of the width)
      "after": "<open id>"    optional, closed questions: condition on that open question's answer}
    """
    if not isinstance(questions, list) or not 1 <= len(questions) <= MAX_QUESTIONS:
        raise ValueError(f"questions must be a list of 1-{MAX_QUESTIONS} questions")
    out, ids = [], set()
    for raw in questions:
        q = {"id": str(raw.get("id", "")).strip(), "kind": raw.get("kind"), "text": str(raw.get("text", "")).strip()}
        if not q["id"] or q["id"] in ids:
            raise ValueError(f"question ids must be unique and non-empty: {q['id']!r}")
        if q["kind"] not in KINDS:
            raise ValueError(f"{q['id']}: kind must be one of {KINDS}")
        if not q["text"]:
            raise ValueError(f"{q['id']}: empty text")
        if q["kind"] == "choice":
            options = [str(o).strip().lower() for o in raw.get("options", [])]
            if not 2 <= len(options) <= 12 or len(set(options)) != len(options) or not all(options):
                raise ValueError(f"{q['id']}: choice needs 2-12 distinct options")
            q["options"] = options
        if q["kind"] == "open":
            q["max_tokens"] = max(1, min(int(raw.get("max_tokens", 32)), MAX_OPEN_TOKENS))
        if raw.get("crop") is not None:
            x0, x1 = (float(v) for v in raw["crop"])
            if not 0.0 <= x0 < x1 <= 1.0:
                raise ValueError(f"{q['id']}: crop must be 0 <= x0 < x1 <= 1")
            q["crop"] = [x0, x1]
        ids.add(q["id"])
        out.append(q)
    by_id = {q["id"]: q for q in out}
    for q, raw in zip(out, questions):
        after = raw.get("after")
        if after is not None:
            if q["kind"] == "open" or by_id.get(after, {}).get("kind") != "open":
                raise ValueError(f"{q['id']}: 'after' must name an open question, on a closed question")
            q["after"] = after
    return out


def clean_answer(text):
    """A generated answer as one trimmed line. Numbered lists ("1. A red ball. 2. A red ball.") become a
    comma-separated list without repeats, since small models often loop on the same item."""
    text = " ".join(text.split())
    items = [i.strip(" .,") for i in re.split(r"(?:^|\s)\d+\.\s", text) if i.strip(" .,")]
    if len(items) > 1 or re.match(r"\d+\.\s", text):
        text = ", ".join(dict.fromkeys(items))
    return text[:400]


def load_image(path):
    return Image.open(path).convert("RGB")
