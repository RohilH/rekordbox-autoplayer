import numpy as np

from autoplayer.audio.output import BlockQueue


def block(value: float, n: int) -> np.ndarray:
    return np.full((n, 2), value, dtype=np.float32)


def test_pull_spans_blocks_and_pads_with_silence():
    queue = BlockQueue()
    queue.push(block(1, 3))
    queue.push(block(2, 3))
    out = queue.pull(8)
    np.testing.assert_array_equal(out[:, 0], [1, 1, 1, 2, 2, 2, 0, 0])
    assert queue.played == 6
    assert queue.queued == 0


def test_reset_keeps_a_fade_out_and_restarts_the_clock():
    queue = BlockQueue()
    queue.push(block(1, 100))
    queue.pull(10)
    queue.reset(fade_frames=5)
    assert queue.queued == 5
    assert queue.played == -5
    out = queue.pull(5)[:, 0]
    assert out[0] == 1 and out[-1] == 0
    assert queue.played == 0
