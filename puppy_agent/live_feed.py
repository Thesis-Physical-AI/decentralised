"""Live camera feed for the robot's VLM: frames stream at a fixed rate and the VLM keeps analysing the newest one.

Instead of stopping for a snapshot and asking one yes/no question, the robot keeps moving while:
  - the camera produces a frame every 1/feed_hz s of robot time (feed_hz >= 1), each one published to the
    VLM server's live view;
  - whenever the VLM is free it takes the newest frame and answers a whole set of questions about it in
    one call (frame_questions): open questions in words (what is in view, which objects, where the target
    is) and closed ones scored as probabilities (is the target visible, which side, is it close, is the way
    blocked, which action from the closed menu);
  - each analysis is released to the executor only once the robot's clock has passed
    frame time + onboard latency, so a slow onboard computer still shows up as stale answers.

Open answers are for logs, the live view and people. They are never parsed to move the robot: the executor
acts only on closed answers, validated against the menu (selector.py).

mode="async": the VLM call runs on a worker thread and the simulation keeps stepping meanwhile (live in
wall-clock time as well). If the VLM answers later in robot time than its onboard latency allows (the
simulation ran faster than the model), the analysis is released when it arrives and counted in
stats()["late"]. mode="lockstep": the simulation waits for each answer, so robot-time timing is exact.
"""
import math
import statistics
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field

from .selector import DESCRIPTIONS

# Prompts shared with vlm_server/smolvlm.py (QUESTION, NEAR_QUESTION, CROPS; kept equal by a unit test),
# so the calibrated thresholds (run_prototype.py THRESHOLDS) apply to the live answers too.
VISIBLE_Q = "Is there {phrase} in this image? Answer with yes or no."
NEAR_Q = "Is {phrase} close to the camera, taking up a large part of the image? Answer with yes or no."
BLOCKED_Q = "Is there an obstacle right in front of the camera, blocking the way forward? Answer with yes or no."
SCENE_Q = "Describe this image in one sentence."
OBJECTS_Q = "What objects are in this image? Name each one with its colour."
TARGET_Q = "Describe where {phrase} is in this image."
ACTION_Q = ("You steer a small four-legged robot. Its task: {instruction}. Its target is {phrase}.\n"
            "Last actions: {history}.\nPossible actions:\n{menu}\nWhich action should it take? "
            "Answer with one action name.")
CROPS = {"left": (0.0, 0.45), "center": (0.275, 0.725), "right": (0.55, 1.0)}
OPEN_IDS = ("scene", "objects", "target")


def frame_questions(phrase, instruction, history=(), open_tokens=32):
    """The questions asked about every analysed frame (server format: vlm_server/smolvlm.py validate_questions)."""
    menu = "\n".join(f"{name} - {meaning}" for name, meaning in DESCRIPTIONS.items())
    history = ", ".join(list(history)[-4:]) or "none"
    questions = [
        {"id": "scene", "kind": "open", "text": SCENE_Q, "max_tokens": open_tokens},
        {"id": "objects", "kind": "open", "text": OBJECTS_Q, "max_tokens": open_tokens},
        {"id": "target", "kind": "open", "text": TARGET_Q.format(phrase=phrase), "max_tokens": open_tokens},
        {"id": "visible", "kind": "yesno", "text": VISIBLE_Q.format(phrase=phrase)},
        {"id": "near", "kind": "yesno", "text": NEAR_Q.format(phrase=phrase)},
        {"id": "blocked", "kind": "yesno", "text": BLOCKED_Q},
    ]
    questions += [{"id": f"where_{side}", "kind": "yesno", "text": VISIBLE_Q.format(phrase=phrase), "crop": list(crop)}
                  for side, crop in CROPS.items()]
    questions.append({"id": "action", "kind": "choice", "options": list(DESCRIPTIONS), "after": "target",
                      "text": ACTION_Q.format(instruction=instruction, phrase=phrase, history=history, menu=menu)})
    return questions


@dataclass
class FrameAnalysis:
    """Every answer about one frame, raw (`answers`) and interpreted with the robot's thresholds."""
    source: str  # "vlm" or "perception"
    latency_s: float  # onboard latency of the whole question set
    answers: dict  # question id -> {"answer": ..., "p_yes" | "scores": ...}
    visible: bool = False
    visible_p: float = 0.0
    where: str = "none"  # "left", "center", "right" or "none"
    near: bool = False
    blocked: bool = False
    action: str = ""  # untrusted: validated by the executor
    action_scores: dict = None
    scene: str = ""
    objects: str = ""
    target: str = ""
    detail: str = ""
    frame_id: int = 0
    frame_t: float = 0.0  # robot time the frame was taken
    age_s: float = 0.0  # robot time from frame to release
    late_s: float = 0.0  # how much later than its onboard latency the answer arrived (async mode)
    image: object = field(default=None, repr=False)  # the analysed frame

    @property
    def confidence(self):
        return (self.action_scores or {}).get(self.action, 0.0)

    def summary(self):
        """Closed answers in one line, for logs."""
        return (f"action={self.action} visible={'yes' if self.visible else 'no'}({self.visible_p:.2f}) "
                f"where={self.where} near={'yes' if self.near else 'no'} blocked={'yes' if self.blocked else 'no'}")


def interpret(answers, source, latency_s, visible_threshold=0.5, near_threshold=0.5, blocked_threshold=0.5,
              where_threshold=None, detail=""):
    """FrameAnalysis from raw answers. Missing answers count as "no" / "none"."""
    where_threshold = visible_threshold if where_threshold is None else where_threshold

    def p(qid):
        return float(answers.get(qid, {}).get("p_yes", 0.0))

    crops = {side: p(f"where_{side}") for side in CROPS}
    best = max(crops, key=crops.get)
    action = answers.get("action", {})
    return FrameAnalysis(
        source, latency_s, answers, visible=p("visible") >= visible_threshold, visible_p=p("visible"),
        where=best if crops[best] >= where_threshold else "none", near=p("near") >= near_threshold,
        blocked=p("blocked") >= blocked_threshold, action=str(action.get("answer", "")),
        action_scores={str(k): float(v) for k, v in action.get("scores", {}).items()},
        **{qid: str(answers.get(qid, {}).get("answer", "")) for qid in OPEN_IDS}, detail=detail)


class LiveFeed:
    def __init__(self, camera, now, analyze, questions, publish=None, feed_hz=2.0, mode="async",
                 charge_latency=True):
        """camera() -> RGB frame; now() -> robot time; analyze(image, questions) -> FrameAnalysis (blocking);
        questions() -> the question list for the next analysis; publish(image, frame_id, t) streams a frame
        to the live view (optional, best effort: frames are dropped while the previous one is still sending)."""
        if feed_hz < 1.0:
            raise ValueError(f"feed_hz must be at least 1 Hz, got {feed_hz}")
        if mode not in ("async", "lockstep"):
            raise ValueError(f"mode must be 'async' or 'lockstep', got {mode!r}")
        self.camera, self.now, self.analyze, self.questions, self.publish = camera, now, analyze, questions, publish
        self.period, self.mode, self.charge_latency = 1.0 / feed_hz, mode, charge_latency
        self._worker = ThreadPoolExecutor(1, "vlm-ask") if mode == "async" else None
        self._sender = ThreadPoolExecutor(1, "vlm-feed") if publish else None
        self._sending = None
        self._next_frame_t = None
        self._latest = None  # (frame_id, t, image)
        self._pending = None  # [frame_id, frame_t, image, future, last robot time it was seen unfinished]
        self._last_submitted = 0
        self.frames, self.analyses, self.late, self.latencies = 0, 0, 0, []
        self.started_at = now()

    def tick(self):
        """Call once per control tick. Captures a frame when one is due, keeps the VLM busy on the newest
        frame, and returns a FrameAnalysis when one is released (else None)."""
        t = self.now()
        if self._next_frame_t is None or t >= self._next_frame_t - 1e-9:
            self._capture(t)
        released = self._release(t)
        if self._pending is None and self._latest and self._latest[0] > self._last_submitted:
            self._submit()
            if released is None:
                released = self._release(t, just_submitted=True)  # lockstep, no latency charge: available at once
        return released

    def _capture(self, t):
        image = self.camera()
        self.frames += 1
        self._latest = (self.frames, t, image)
        self._next_frame_t = t + self.period if self._next_frame_t is None else self._next_frame_t + self.period
        if self._next_frame_t <= t:
            self._next_frame_t = t + self.period  # fell behind (long tick): don't burst
        if self._sender and (self._sending is None or self._sending.done()):
            self._sending = self._sender.submit(self.publish, image, self.frames, t)

    def _submit(self):
        frame_id, frame_t, image = self._latest
        questions = self.questions()
        if self._worker:
            future = self._worker.submit(self.analyze, image, questions)
        else:
            future = Future()
            future.set_result(self.analyze(image, questions))
        self._pending = [frame_id, frame_t, image, future, None]
        self._last_submitted = frame_id

    def _release(self, t, just_submitted=False):
        if self._pending is None:
            return None
        if not self._pending[3].done():
            if not just_submitted:
                self._pending[4] = t
            return None
        frame_id, frame_t, image, future, unfinished_t = self._pending
        analysis = future.result()  # re-raises a failed request
        ready_t = frame_t + (analysis.latency_s if self.charge_latency else 0.0)
        if t < ready_t - 1e-9:
            return None  # the onboard computer would still be thinking
        self._pending = None
        analysis.frame_id, analysis.frame_t, analysis.image = frame_id, frame_t, image
        analysis.age_s = t - frame_t
        # late only if the answer was still missing at a tick at or after its due time
        analysis.late_s = max(0.0, unfinished_t - ready_t) if unfinished_t is not None and unfinished_t >= ready_t else 0.0
        self.analyses += 1
        self.late += unfinished_t is not None and unfinished_t >= ready_t
        self.latencies.append(analysis.latency_s)
        return analysis

    def stats(self):
        """Rates in robot time over the feed's lifetime."""
        span = max(self.now() - self.started_at, 1e-9)
        return {"frames": self.frames, "analyses": self.analyses, "robot_seconds": round(span, 2),
                "frame_hz": round(self.frames / span, 2), "analysis_hz": round(self.analyses / span, 2),
                "median_latency_s": round(statistics.median(self.latencies), 3) if self.latencies else None,
                "late": self.late}

    def close(self):
        for pool in (self._worker, self._sender):
            if pool:
                pool.shutdown(wait=True, cancel_futures=True)


@dataclass
class Rotation:
    """Accumulates |yaw change| (rad), for "have I looked all the way round?"."""
    total: float = 0.0
    _last: float = field(default=None, repr=False)

    def update(self, yaw):
        if self._last is not None:
            self.total += abs(math.remainder(yaw - self._last, 2 * math.pi))
        self._last = yaw
        return self.total

    def reset(self):
        self.total, self._last = 0.0, None
        return 0.0
