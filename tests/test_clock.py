import pytest

from autoplayer.audio.clock import MixClock


def test_tau_follows_speed_changes():
    clock = MixClock()
    clock.set(100, speed=2.0)
    assert clock.tau(50) == pytest.approx(50)
    assert clock.tau(150) == pytest.approx(200)
    assert clock.out_time(200) == pytest.approx(150)


def test_a_jump_keeps_the_latest_speed_even_if_it_was_scheduled_later():
    clock = MixClock()
    clock.set(1000, speed=1.5)
    clock.set(400, tau=10_000)
    assert clock.speed_at(400) == 1.5
    assert clock.tau(500) == pytest.approx(10_150)


def test_setting_earlier_replaces_later_segments():
    clock = MixClock()
    clock.set(1000, speed=2.0)
    clock.set(500, speed=0.5)
    assert clock.tau(1500) == pytest.approx(500 + 0.5 * 1000)
