# Tests of the balanced sampling of wm2 training (A1 + A2 of docs/data_solve.md).
#   venv\Scripts\python -m pytest autoresearch/tests/test_balance.py -q
import numpy as np
import pytest
import torch

from wm2.data import SPEED_MAX, draw_order, sample_weights


def fake_data(track, source, speed):
    n = len(track)
    return {"state": torch.tensor(np.column_stack([np.array(speed) / SPEED_MAX, np.zeros(n)]), dtype=torch.float32),
            "track": np.array(track), "source": np.array(source)}


def test_each_track_gets_half_of_the_draws():
    data = fake_data(["practice"] * 80 + ["compete"] * 20, ["random_practice"] * 80 + ["fast_compete"] * 20, [1.0] * 100)
    w = sample_weights(data, torch.arange(100))
    assert w[:80].sum() == pytest.approx(0.5) and w[80:].sum() == pytest.approx(0.5)


def test_standing_fast_and_new_trips_are_weighted():
    data = fake_data(["compete"] * 4, ["fast_compete", "fast_compete", "fast_compete", "self4_compete"],
                     [1.0, 0.1, 2.0, 1.0])
    w = sample_weights(data, torch.arange(4)).numpy()
    assert w[1] / w[0] == pytest.approx(0.2)      # standing: 1 in 5
    assert w[2] / w[0] == pytest.approx(2.0)      # fast: x2
    assert w[3] / w[0] == pytest.approx(2.0)      # newest compete trips: x2


def test_old_data_without_track_is_refused():
    with pytest.raises(SystemExit):
        sample_weights({"state": torch.zeros(3, 2)}, torch.arange(3))


def test_draw_order_keeps_the_epoch_size():
    starts = torch.arange(10, 20)
    order = draw_order(starts, torch.full((10,), 0.1, dtype=torch.float64))
    assert len(order) == 10 and set(order.tolist()) <= set(range(10, 20))
