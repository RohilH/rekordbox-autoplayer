import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from autoplayer.audio import engine as engine_module
from autoplayer.audio.decode import SAMPLE_RATE
from autoplayer.library.pdb import Track
from autoplayer.library.anlz import Cue
from tests.helpers import click_track

BPMS = [124, 126, 128]


class FakeOutput:
    def __init__(self, queue, device=None):
        self.queue = queue
        self.paused = False

    device_name = "Fake output"

    def start(self):
        pass

    def close(self):
        pass

    def play(self, seconds: float, timeout: float = 10.0) -> np.ndarray:
        """Consume `seconds` of rendered audio, waiting for the producer when the queue runs dry."""
        remaining = int(seconds * SAMPLE_RATE)
        deadline = time.monotonic() + timeout
        heard = []
        while remaining > 0:
            if time.monotonic() > deadline:
                raise AssertionError("producer stalled")
            available = min(self.queue.queued, remaining)
            if available:
                heard.append(self.queue.pull(available))
                remaining -= available
            else:
                time.sleep(0.005)
        return np.concatenate(heard) if heard else np.zeros((0, 2), dtype=np.float32)


def fake_track(i: int) -> Track:
    return Track(i, f"Track {i}", "Artist", "", "8A", BPMS[i], 60, Path(f"/nowhere/{i}.mp3"), None)


@pytest.fixture
def engine(monkeypatch):
    monkeypatch.setattr(engine_module, "AudioOutput", FakeOutput)
    monkeypatch.setattr(
        engine_module,
        "load_track",
        lambda track, index: click_track(track.bpm, 150, cues=(Cue(time_s=20.0, hot_cue=0, is_loop=False),), index=index),
    )
    eng = engine_module.Engine()
    eng.start()
    yield eng
    eng.close()


def wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition not reached")


def test_plays_the_playlist_in_order_and_plans_beatmatched_mixes(engine):
    engine.play([fake_track(i) for i in range(3)], 0, playlist_id=7, playlist_name="Test")
    wait_for(lambda: engine.queue.queued > 0)
    engine.output.play(0.1)
    wait_for(lambda: engine.state().get("next_index") == 1)

    state = engine.state()
    assert state["index"] == 0
    assert state["track"]["title"] == "Track 0"
    assert state["next_beatmatched"] is True
    assert state["mix_in_s"] > 0


def test_skip_moves_to_the_next_track(engine):
    engine.play([fake_track(i) for i in range(3)], 0)
    wait_for(lambda: engine.queue.queued > 0)
    engine.output.play(0.1)
    engine.skip()
    wait_for(lambda: engine.queue.queued > SAMPLE_RATE // 10)
    engine.output.play(0.2)
    wait_for(lambda: engine.state().get("index") == 1)


def test_skipping_past_the_last_track_stops(engine):
    engine.play([fake_track(0)], 0)
    wait_for(lambda: engine.queue.queued > 0)
    engine.skip()
    assert engine.state()["status"] == "idle"


def started(engine, tracks=3):
    engine.play([fake_track(i) for i in range(tracks)], 0)
    wait_for(lambda: engine.queue.queued > 0)
    engine.output.play(0.1)
    wait_for(lambda: engine.state().get("next_index") == 1)
    return engine.state()


def settle(engine):
    engine.output.play(SKIP_FADE_SECONDS)
    return engine.state()


SKIP_FADE_SECONDS = engine_module.SKIP_FADE_FRAMES / SAMPLE_RATE + 0.01


def test_seek_within_a_track_jumps_there(engine):
    started(engine)
    engine.seek(20.0)
    state = settle(engine)
    assert state["index"] == 0
    assert state["position_s"] == pytest.approx(20.0, abs=0.05)
    assert not state["mixing"]


def test_seek_into_the_outgoing_transition_plays_both_tracks(engine):
    state = started(engine)
    target = state["mix_out_start_s"] + 2
    engine.seek(target)
    state = settle(engine)
    assert state["mixing"]
    assert state["index"] == 0
    assert state["position_s"] == pytest.approx(target, abs=0.05)


def test_seek_back_into_the_incoming_transition_plays_both_tracks(engine):
    state = started(engine)
    engine.seek(state["play_end_s"] - 0.5)
    state = settle(engine)
    engine.output.play(1.0)
    state = engine.state()
    assert state["index"] == 1 and not state["mixing"]

    target = state["play_start_s"] + 1
    engine.seek(target)
    state = settle(engine)
    assert state["mixing"]
    assert state["index"] == 1
    assert state["position_s"] == pytest.approx(target, abs=SKIP_FADE_SECONDS + 0.01)


def test_seek_before_the_mix_in_point_restarts_the_track_alone(engine):
    state = started(engine)
    engine.seek(state["play_end_s"] - 0.5)
    settle(engine)
    engine.output.play(1.0)
    state = engine.state()
    target = state["play_start_s"] - 5
    assert target > 0

    engine.seek(target)
    wait_for(lambda: engine.state().get("index") == 1)
    state = settle(engine)
    assert state["index"] == 1
    assert not state["mixing"]
    assert state["position_s"] == pytest.approx(target, abs=SKIP_FADE_SECONDS + 0.01)


def test_state_stays_json_encodable_through_a_transition(engine):
    state = started(engine)
    engine.seek(state["mix_out_start_s"] + 1)
    settle(engine)
    for _ in range(6):
        json.dumps(engine.state())
        engine.output.play(5.0)
    state = engine.state()
    json.dumps(state)
    assert state["index"] == 1


def test_master_bpm_speeds_up_the_current_track_without_a_jump(engine):
    started(engine)
    engine.output.play(2.0)
    before = engine.state()
    engine.set_master_bpm(130)
    engine.output.play(4.0)
    after = engine.state()
    assert after["bpm"] == pytest.approx(130, abs=0.01)
    assert after["master_bpm"] == 130
    expected = before["position_s"] + 0.25 + (4.0 - 0.25) * 130 / 124
    assert after["position_s"] == pytest.approx(expected, abs=0.1)
    assert after["key_locked"]


def test_master_bpm_holds_through_the_next_transition(engine):
    state = started(engine)
    engine.set_master_bpm(122)
    settle(engine)
    wait_for(lambda: engine.state().get("mix_out_start_s") is not None)
    state = engine.state()
    engine.seek(state["play_end_s"] - 0.5)
    settle(engine)
    engine.output.play(1.0)
    state = engine.state()
    assert state["index"] == 1
    assert state["bpm"] == pytest.approx(122, abs=0.05)


def test_master_bpm_can_go_far_from_the_tracks_own_tempo(engine):
    started(engine)
    engine.set_master_bpm(170)
    engine.output.play(2.0)
    assert engine.state()["bpm"] == pytest.approx(170, abs=0.01)


def test_resetting_master_bpm_returns_to_the_tracks_tempo(engine):
    started(engine)
    engine.set_master_bpm(130)
    engine.output.play(1.0)
    engine.set_master_bpm(None)
    engine.output.play(1.0)
    state = engine.state()
    assert state["master_bpm"] is None
    assert state["bpm"] == pytest.approx(124, abs=0.01)
    assert not state["key_locked"]


def test_master_bpm_keeps_the_pitch_with_key_lock(monkeypatch, engine):
    def toned(track, index):
        loaded = click_track(track.bpm, 150, index=index)
        t = np.arange(loaded.audio.shape[0]) / SAMPLE_RATE
        loaded.audio = (loaded.audio * 0 + 0.3 * np.sin(2 * np.pi * 440 * t)[:, None]).astype(np.float32)
        return loaded

    monkeypatch.setattr(engine_module, "load_track", toned)
    started(engine)
    engine.set_master_bpm(136)
    engine.output.play(1.0)
    heard = engine.output.play(2.0)[:, 0]
    spectrum = np.abs(np.fft.rfft(heard * np.hanning(len(heard))))
    peak = np.fft.rfftfreq(len(heard), 1 / SAMPLE_RATE)[spectrum.argmax()]
    assert peak == pytest.approx(440, abs=2)


def test_seeking_works_while_the_next_track_is_still_loading(monkeypatch, engine):
    release = threading.Event()

    def slow_after_first(track, index):
        if index > 0:
            release.wait(10)
        return click_track(track.bpm, 150, index=index)

    monkeypatch.setattr(engine_module, "load_track", slow_after_first)
    engine.play([fake_track(i) for i in range(3)], 0)
    wait_for(lambda: engine.queue.queued > 0)
    engine.output.play(0.1)
    started_at = time.monotonic()
    engine.seek(100.0)
    assert time.monotonic() - started_at < 0.5
    state = settle(engine)
    assert state["position_s"] == pytest.approx(100.0, abs=0.1)
    release.set()


def test_tracks_coming_in_without_a_beat_match_hold_the_master_bpm(engine):
    engine.update_settings(mode="off")
    started(engine)
    engine.set_master_bpm(122)
    settle(engine)
    state = engine.state()
    wait_for(lambda: engine._session and len(engine._session.timeline.placements) > 1)
    engine.seek(engine.state()["play_end_s"] - 0.5)
    settle(engine)
    engine.output.play(1.0)
    state = engine.state()
    assert state["index"] == 1
    assert state["bpm"] == pytest.approx(122, abs=0.05)


def other_track(track_id: int, bpm: float = 125) -> Track:
    return Track(track_id, f"Queued {track_id}", "Elsewhere", "", "5A", bpm, 150, Path(f"/nowhere/q{track_id}.mp3"), None)


def planned_titles(engine) -> list[str]:
    session = engine._session
    return [session.sequence[p.item.index].track.title for p in session.timeline.placements]


def test_a_queued_track_is_mixed_in_next_and_the_playlist_resumes_after_it(engine):
    started(engine)
    assert planned_titles(engine) == ["Track 0", "Track 1"]
    engine.queue_add(other_track(10), "Other playlist")
    wait_for(lambda: planned_titles(engine) == ["Track 0", "Queued 10"])
    state = engine.state()
    assert state["next_track"]["title"] == "Queued 10" and state["next_from_queue"]
    assert [q["track"]["title"] for q in state["up_next"]] == ["Queued 10"]

    engine.seek(state["play_end_s"] - 0.5)
    settle(engine)
    engine.output.play(1.0)
    state = engine.state()
    assert state["track"]["title"] == "Queued 10" and state["from_queue"]
    assert state["up_next"] == []
    wait_for(lambda: engine.state().get("next_track", {}).get("title") == "Track 1")


def test_removing_the_planned_queued_track_puts_the_playlist_back(engine):
    started(engine)
    queued = engine.queue_add(other_track(10))
    wait_for(lambda: planned_titles(engine)[-1] == "Queued 10")
    engine.queue_remove(queued.id)
    wait_for(lambda: planned_titles(engine)[-1] == "Track 1")


def test_reordering_the_queue_changes_what_comes_next(engine):
    started(engine)
    engine.queue_add(other_track(10))
    later = engine.queue_add(other_track(11))
    wait_for(lambda: planned_titles(engine)[-1] == "Queued 10")
    engine.queue_move(later.id, 0)
    wait_for(lambda: planned_titles(engine)[-1] == "Queued 11")
    assert [q["track"]["title"] for q in engine.state()["up_next"]] == ["Queued 11", "Queued 10"]


def test_adding_to_the_end_of_the_queue_keeps_the_planned_transition(engine):
    started(engine)
    engine.queue_add(other_track(10))
    wait_for(lambda: planned_titles(engine)[-1] == "Queued 10")
    planned = engine._session.timeline.placements[-1]
    engine.queue_add(other_track(11))
    assert engine._session.timeline.placements[-1] is planned


def test_skip_goes_to_the_queue_and_the_queue_survives_it(engine):
    started(engine)
    engine.queue_add(other_track(10))
    engine.queue_add(other_track(11))
    engine.skip()
    wait_for(lambda: engine.state().get("track", {}).get("title") == "Queued 10")
    engine.output.play(0.2)
    assert [q["track"]["title"] for q in engine.state()["up_next"]] == ["Queued 11"]
    wait_for(lambda: engine.state().get("next_track", {}).get("title") == "Queued 11")
    engine.queue_clear()
    wait_for(lambda: engine.state().get("next_track", {}).get("title") == "Track 1")


def test_a_sorted_playlist_plays_in_sorted_order_and_reports_playlist_positions(engine):
    tracks = [fake_track(i) for i in range(3)]
    order = [2, 0, 1]
    engine.play([tracks[i] for i in order], 0, positions=order)
    wait_for(lambda: engine.queue.queued > 0)
    engine.output.play(0.1)
    wait_for(lambda: engine.state().get("next_track") is not None)
    state = engine.state()
    assert state["track"]["title"] == "Track 2"
    assert state["index"] == 2
    assert state["next_track"]["title"] == "Track 0"
    assert state["next_index"] == 0
    assert [t["title"] for t in state["then_tracks"]] == ["Track 0", "Track 1"]
