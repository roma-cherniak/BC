# Tests of the recovery pushes (wm2/push.py, B4 of docs/data_solve.md).
#   venv\Scripts\python -m pytest autoresearch/tests/test_push.py -q
import random

from wm2.push import PUSH_EVERY, PUSH_TIME, new_push_state, push_steer


def drive(seconds, seed=0, dt=0.05):
    rng, state, out = random.Random(seed), None, []
    state = new_push_state(0.0, rng)
    for k in range(int(seconds / dt)):
        state, steer = push_steer(state, k * dt, rng)
        out.append(steer)
    return out


def test_no_push_before_the_first_wait():
    steers = drive(PUSH_EVERY[0] - 0.1)
    assert all(s is None for s in steers)


def test_pushes_are_full_lock_and_short():
    steers = drive(300)
    pushes, run = [], 0
    for s in steers + [None]:
        if s is None:
            if run:
                pushes.append(run * 0.05)
            run = 0
        else:
            assert s in (-1.0, 1.0)
            run += 1
    assert len(pushes) >= 300 / (PUSH_EVERY[1] + PUSH_TIME[1]) - 1
    assert all(PUSH_TIME[0] - 0.06 <= p <= PUSH_TIME[1] + 0.06 for p in pushes)


def test_share_of_push_time_is_small():
    steers = drive(600)
    share = sum(s is not None for s in steers) / len(steers)
    assert 0.02 < share < 0.12


def test_state_is_not_changed_in_place():
    rng = random.Random(1)
    state = new_push_state(0.0, rng)
    before = dict(state)
    push_steer(state, 100.0, rng)
    assert state == before
