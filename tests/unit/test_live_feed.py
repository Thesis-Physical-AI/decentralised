"""Live camera feed: frame rate, latency charging, newest-frame analysis, the question set, and live search."""
import ast
import math
import os
import re
import threading
import time

import pytest

from puppy_agent import contracts as c
from puppy_agent.live_feed import CROPS, NEAR_Q, VISIBLE_Q, FrameAnalysis, LiveFeed, frame_questions, interpret

from .fakes import KinematicRobot
from .test_executor import WORLD, step
from .test_live_selector import live_executor
from .test_visual_approach import ground_truth

SMOLVLM = os.path.join(os.path.dirname(__file__), "..", "..", "vlm_server", "smolvlm.py")


def server_definitions():
    """Names from vlm_server/smolvlm.py without importing it (it needs torch)."""
    source = open(SMOLVLM, encoding="utf-8").read()
    tree = ast.parse(source)
    found = {"re": re}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try:
                found[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Tuple):
            for target, value in zip(node.targets[0].elts, node.value.elts):
                found[target.id] = ast.literal_eval(value)
        elif isinstance(node, ast.FunctionDef) and node.name in ("validate_questions", "clean_answer"):
            exec(ast.get_source_segment(source, node), found)
    return found


class Clock:
    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t


def fixed(latency_s=0.0, wall_s=0.0, calls=None):
    def analyze(image, questions):
        if calls is not None:
            calls.append(image)
        if wall_s:
            time.sleep(wall_s)
        return FrameAnalysis("vlm", latency_s, {})
    return analyze


def run(feed, clock, seconds, dt=0.05):
    released = []
    for _ in range(round(seconds / dt)):
        analysis = feed.tick()
        if analysis is not None:
            released.append((clock.t, analysis))
        clock.t += dt
    return released


def frame_counter():
    count = [0]

    def camera():
        count[0] += 1
        return count[0]  # the "image" is the frame number
    return camera


# -- the question set ------------------------------------------------------------------------------
def test_question_set_is_valid_for_the_server_and_mixes_open_and_closed():
    server = server_definitions()
    questions = frame_questions("a red ball", "[movetowards] apple", ["search", "left"])
    checked = server["validate_questions"](questions)
    kinds = {q["kind"] for q in checked}
    assert kinds == {"open", "yesno", "choice"}
    assert sum(q["kind"] == "open" for q in checked) >= 2
    assert len(checked) <= server["MAX_QUESTIONS"]
    action = next(q for q in checked if q["id"] == "action")
    assert action["after"] == "target" and "search, left" in action["text"]


def test_live_prompts_match_the_calibrated_server_prompts():
    server = server_definitions()
    assert VISIBLE_Q == server["QUESTION"] and NEAR_Q == server["NEAR_QUESTION"]
    assert {k: tuple(v) for k, v in CROPS.items()} == server["CROPS"]


@pytest.mark.parametrize("bad", [
    [],
    [{"id": "a", "kind": "maybe", "text": "?"}],
    [{"id": "a", "kind": "choice", "text": "?", "options": ["only"]}],
    [{"id": "a", "kind": "yesno", "text": "?"}, {"id": "a", "kind": "yesno", "text": "?"}],
    [{"id": "a", "kind": "yesno", "text": "?", "crop": [0.6, 0.2]}],
    [{"id": "a", "kind": "yesno", "text": "?", "after": "a"}],
])
def test_server_rejects_malformed_questions(bad):
    with pytest.raises(ValueError):
        server_definitions()["validate_questions"](bad)


def test_server_caps_open_answer_length():
    q = server_definitions()["validate_questions"]([{"id": "a", "kind": "open", "text": "?", "max_tokens": 999}])
    assert q[0]["max_tokens"] == 64


def test_repeated_list_answers_are_collapsed():
    clean = server_definitions()["clean_answer"]
    assert clean("1. Blue cube 2. Blue cube 3. White lines") == "Blue cube, White lines"
    assert clean("A red ball on   a\nbrown box.") == "A red ball on a brown box."


def test_interpret_applies_thresholds_and_keeps_words():
    answers = {"visible": {"p_yes": 0.02}, "near": {"p_yes": 0.4}, "blocked": {"p_yes": 0.9},
               "where_left": {"p_yes": 0.01}, "where_center": {"p_yes": 0.03}, "where_right": {"p_yes": 0.005},
               "action": {"answer": "forward", "scores": {"forward": 0.7, "left": 0.3}},
               "scene": {"answer": "A brown box on a blue floor."}}
    a = interpret(answers, "vlm", 0.3, visible_threshold=0.0151, near_threshold=0.32)
    assert a.visible and a.near and a.blocked and a.where == "center"
    assert a.action == "forward" and a.confidence == 0.7 and a.scene.startswith("A brown box")
    assert interpret(answers, "vlm", 0.3, visible_threshold=0.5).where == "none"


# -- the feed --------------------------------------------------------------------------------------
def test_feed_rate_below_1hz_is_refused():
    with pytest.raises(ValueError):
        LiveFeed(lambda: 0, lambda: 0.0, fixed(), lambda: [], feed_hz=0.5)


def test_frames_stream_at_feed_rate_and_fast_model_sees_every_frame():
    clock = Clock()
    feed = LiveFeed(frame_counter(), clock.now, fixed(), lambda: [], feed_hz=2.0, mode="lockstep")
    released = run(feed, clock, 10.0)
    stats = feed.stats()
    assert stats["frames"] == 20 and stats["frame_hz"] >= 1.0
    assert len(released) == 20 and stats["analysis_hz"] >= 1.0
    assert [a.frame_id for _, a in released] == list(range(1, 21))


def test_slow_model_answers_late_and_always_takes_the_newest_frame():
    clock, calls = Clock(), []
    feed = LiveFeed(frame_counter(), clock.now, fixed(latency_s=1.3, calls=calls), lambda: [], feed_hz=4.0,
                    mode="lockstep")
    released = run(feed, clock, 10.0)
    assert feed.stats()["frames"] == 40
    for t, analysis in released:
        assert t >= analysis.frame_t + 1.3 - 1e-9  # never before the onboard computer could have answered
        assert analysis.age_s == pytest.approx(t - analysis.frame_t)
    ids = [a.frame_id for _, a in released]
    assert all(b - a >= 5 for a, b in zip(ids, ids[1:]))  # frames in between are skipped, not queued
    assert calls[:len(ids)] == ids  # the frame analysed is the frame reported (one more may be in flight)


def test_async_feed_keeps_the_clock_running_while_the_model_thinks():
    clock, worker_threads = Clock(), set()

    def analyze(image, questions):
        worker_threads.add(threading.current_thread().name)
        time.sleep(0.02)
        return FrameAnalysis("vlm", 0.5, {})

    feed = LiveFeed(frame_counter(), clock.now, analyze, lambda: [], feed_hz=2.0, mode="async")
    released = []
    deadline = time.monotonic() + 5
    while len(released) < 3 and time.monotonic() < deadline:
        analysis = feed.tick()
        if analysis is not None:
            released.append(analysis)
        clock.t += 0.01
        time.sleep(0.002)
    feed.close()
    assert len(released) == 3
    assert threading.current_thread().name not in worker_threads
    assert all(a.age_s >= 0.5 - 1e-9 for a in released)


def test_answers_slower_than_their_onboard_latency_are_counted_late():
    clock = Clock()
    feed = LiveFeed(frame_counter(), clock.now, fixed(latency_s=0.0, wall_s=0.05), lambda: [], feed_hz=2.0)
    released = []
    deadline = time.monotonic() + 5
    while not released and time.monotonic() < deadline:
        analysis = feed.tick()
        if analysis is not None:
            released.append(analysis)
        clock.t += 0.05  # the sim runs ahead of the model
        time.sleep(0.005)
    feed.close()
    assert released and released[0].late_s > 0 and feed.stats()["late"] == 1


def test_fast_answers_are_not_counted_late():
    clock = Clock()
    feed = LiveFeed(frame_counter(), clock.now, fixed(latency_s=0.0), lambda: [], feed_hz=2.0)
    released = []
    deadline = time.monotonic() + 5
    while len(released) < 3 and time.monotonic() < deadline:
        analysis = feed.tick()
        if analysis is not None:
            released.append(analysis)
        time.sleep(0.01)  # the answer arrives between ticks, before the clock moves on
        clock.t += 0.25
    feed.close()
    assert len(released) == 3 and feed.stats()["late"] == 0


def test_frames_are_published_to_the_live_view_without_blocking():
    """Each frame goes to the live view; if a send is still running the frame is dropped, never queued."""
    clock, published = Clock(), []
    feed = LiveFeed(frame_counter(), clock.now, fixed(), lambda: [], feed_hz=2.0, mode="lockstep",
                    publish=lambda image, frame_id, t: published.append((frame_id, t)))
    for _ in range(60):
        feed.tick()
        clock.t += 0.05
        time.sleep(0.003)  # sending is on a thread; give it time, as the sim's real tick does
    feed.close()
    assert [f for f, _ in published] == list(range(1, 7))
    assert [t for _, t in published] == pytest.approx([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])


# -- executor on the feed --------------------------------------------------------------------------
def test_live_approach_records_open_and_closed_answers_for_each_frame():
    robot, events = KinematicRobot(), []
    executor = live_executor(robot, ground_truth(robot), events)
    result = executor.execute(step("movetowards", {"target": "table"}), "puppypi_1")
    assert result.status == c.SUCCEEDED, result.reason
    analyses = [q for q in executor.trace.queries if q["question"] == "analyze"]
    assert analyses and all({"action", "visible", "where", "near", "blocked", "scene", "objects", "target"} <= set(q)
                            for q in analyses)
    feed = next(e for e in events if e["event"] == "feed")
    assert feed["frame_hz"] >= 1.0 and feed["analysis_hz"] >= 1.0
    assert executor.feed_stats == {k: v for k, v in feed.items() if k not in ("step_id", "event")}


def test_live_search_finds_object_and_confirms_it_on_a_still_frame():
    robot, events = KinematicRobot(yaw=math.pi), []  # the apple is behind the robot
    executor = live_executor(robot, ground_truth(robot), events)
    result = executor.execute(step("search", {"object": "apple", "area": "living_room"}), "puppypi_1")
    assert result.status == c.SUCCEEDED, result.reason
    assert result.observations[0]["kind"] == "entity_located"
    assert abs(math.remainder(robot.pose.yaw, 2 * math.pi)) < math.radians(45)  # it turned to face the apple


def test_live_search_reports_absent_object_after_a_full_turn():
    world = {"entities": {**WORLD["entities"], "ghost": {"kind": "object", "description": "a ghost", "in": "living_room"}}}
    robot = KinematicRobot()
    executor = live_executor(robot, ground_truth(robot, world))
    executor.map = type(executor.map).from_world(world)
    result = executor.execute(step("search", {"object": "ghost", "area": "living_room"}), "puppypi_1")
    assert (result.status, result.reason_code) == (c.BLOCKED, "OBJECT_NOT_FOUND")
    assert result.observations[0]["kind"] == "entity_absent"


def test_live_search_does_not_turn_past_a_view_the_model_has_not_seen():
    """With a slow model, every heading still gets an analysis of a frame taken there."""
    world = {"entities": {**WORLD["entities"], "ghost": {"kind": "object", "description": "a ghost", "in": "living_room"}}}
    robot = KinematicRobot()
    perceiver = ground_truth(robot, world)
    slow = perceiver.analyze

    def analyze(image, questions, entity):
        analysis = slow(image, questions, entity)
        analysis.latency_s = 3.0
        return analysis

    perceiver.analyze = analyze
    executor = live_executor(robot, perceiver)
    executor.map = type(executor.map).from_world(world)
    result = executor.execute(step("search", {"object": "ghost", "area": "living_room"}, timeout_s=300), "puppypi_1")
    assert result.reason_code == "OBJECT_NOT_FOUND"
    assert len(executor.trace.queries) >= executor.config.search_turns
