# Code check: small tests with assert. If one of them fails, the numbers of the other checks can't be trusted.
# Run from the BC folder:  venv\Scripts\python -m checks.check_code

import sys
import numpy as np
import torch
from core.models import LATENT, LIDAR, load_run, run_input
from core.sigreg import sigreg
from core.data import load_pairs, pair_obs
from config import FACTS, LIDAR_BEAMS, MAX_RANGE, DEVICE, CURRENT_RUN
from core.sensors import clean_lidar


ROWS = 8                # how many test moments go through the models
CLOUD = 256             # points in the sigreg test clouds (the same as in sigreg.py)


# every small check gets the same things; obs = what the encoder eats (lidar or photos, see core.data.pair_obs)

def check_shapes(encoder, predictor, zond, test, obs):
    _, action, speed, _ = test
    with torch.no_grad():
        z = encoder(obs[:ROWS])
        z_next = predictor(z, action[:ROWS], speed[:ROWS])
    assert z.shape == (ROWS, LATENT), f"encoder gives {tuple(z.shape)}, need {(ROWS, LATENT)}"
    assert z_next.shape == (ROWS, LATENT), f"predictor gives {tuple(z_next.shape)}, need {(ROWS, LATENT)}"


def check_probe(encoder, predictor, zond, test, obs):
    with torch.no_grad():
        facts = zond(encoder(obs[:ROWS]))
    assert facts.shape == (ROWS, len(FACTS)), f"probe gives {facts.shape[1]} numbers, FACTS has {len(FACTS)}"


def check_sigreg(encoder, predictor, zond, test, obs):
    device = test[0].device
    gpus = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=gpus):        # sigreg uses torch.randn: the global dice stay as they were
        torch.manual_seed(0)
        good = sigreg(torch.randn(CLOUD, LATENT, device=device)).item()
        bad = sigreg(torch.zeros(CLOUD, LATENT, device=device)).item()
    assert good < 3, f"sigreg on a perfect cloud = {good:.2f}, should be about 1"
    assert bad > 50, f"sigreg on a collapsed cloud = {bad:.2f}, should be big (about 100)"


def check_no_nan(encoder, predictor, zond, test, obs):
    for name, x in zip(["lidar_now", "action", "speed", "lidar_next", "obs"], [*test, obs]):
        assert torch.isfinite(x).all(), f"{name} has NaN or inf"


def check_clean_lidar(encoder, predictor, zond, test, obs):
    scan = np.linspace(0, 2 * MAX_RANGE, LIDAR_BEAMS)      # 0 .. 20 m: some walls are farther than MAX_RANGE
    scan[4] = np.inf                                        # the game sends inf when a beam hits nothing
    one = clean_lidar(scan)                                 # one scan, like the driver
    table = clean_lidar(np.stack([scan, scan]))             # a table of scans, like prepare_data
    assert one.shape == (LIDAR,), f"one scan: {LIDAR_BEAMS} beams -> {one.shape}, need ({LIDAR},)"
    assert table.shape == (2, LIDAR), f"table of 2 scans -> {table.shape}, need (2, {LIDAR})"
    assert 0 <= one.min() and one.max() <= 1, f"values {one.min():.2f} .. {one.max():.2f}, need 0..1"


SMALL_CHECKS = [check_shapes, check_probe, check_sigreg, check_no_nan, check_clean_lidar]


def check_code(encoder, predictor, zond, test, obs):
    ok = True
    for small_check in SMALL_CHECKS:
        try:
            small_check(encoder, predictor, zond, test, obs)
            print(f"OK    {small_check.__name__}")
        except AssertionError as error:
            print(f"FAIL  {small_check.__name__}: {error}")
            ok = False
    return {"code_ok": ok}


if __name__ == "__main__":
    run = sys.argv[1] if len(sys.argv) > 1 else CURRENT_RUN
    test = load_pairs("test")
    obs = pair_obs("test", run_input(run))
    encoder, predictor, zond = load_run(run, DEVICE)
    print(check_code(encoder, predictor, zond, test, obs))
