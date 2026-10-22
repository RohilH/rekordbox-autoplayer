import pytest

from autoplayer.library.anlz import EMPTY_ANALYSIS, load_analysis
from tests.helpers import anlz_bytes


def test_reads_beat_grid_and_memory_cues(tmp_path):
    path = tmp_path / "ANLZ0000.DAT"
    path.write_bytes(
        anlz_bytes(
            beats=[(1, 12800, 100), (2, 12800, 569), (3, 12800, 1038), (4, 12800, 1506)],
            cues=[(0, 1, 30100), (1, 1, 1000), (0, 2, 60000)],
        )
    )
    analysis = load_analysis(path)

    assert [b.bar_pos for b in analysis.beats] == [1, 2, 3, 4]
    assert analysis.beats[1].time_s == pytest.approx(0.569)
    assert analysis.beats[0].bpm == pytest.approx(128.0)
    assert [(c.time_s, c.hot_cue, c.is_loop) for c in analysis.cues] == [
        (1.0, 1, False),
        (30.1, 0, False),
        (60.0, 0, True),
    ]
    assert [c.time_s for c in analysis.memory_cues] == [30.1]


def test_missing_file_has_no_analysis(tmp_path):
    assert load_analysis(tmp_path / "nope.DAT") is EMPTY_ANALYSIS
    assert load_analysis(None) is EMPTY_ANALYSIS
