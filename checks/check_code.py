# Code check: small tests with assert. If one of them fails, the numbers of the other checks can't be trusted.
# Run from the BC folder:  venv\Scripts\python -m checks.check_code

import torch
from jepa.jepa_model import LATENT
from jepa.sigreg import sigreg
from utils import load_pairs, load_jepa, load_zond, FACTS


ROWS = 8                # how many test moments go through the models
CLOUD = 256             # points in the sigreg test clouds (the same as in sigreg.py)


def check_shapes(encoder, predictor, zond, test):
    lidar_now, action, speed, _ = test
    with torch.no_grad():
        z = encoder(lidar_now[:ROWS].float())
        z_next = predictor(z, action[:ROWS], speed[:ROWS])
    assert z.shape == (ROWS, LATENT), f"encoder gives {tuple(z.shape)}, need {(ROWS, LATENT)}"
    assert z_next.shape == (ROWS, LATENT), f"predictor gives {tuple(z_next.shape)}, need {(ROWS, LATENT)}"


def check_probe(encoder, predictor, zond, test):
    with torch.no_grad():
        facts = zond(encoder(test[0][:ROWS].float()))
    assert facts.shape == (ROWS, len(FACTS)), f"probe gives {facts.shape[1]} numbers, FACTS has {len(FACTS)}"


def check_sigreg(encoder, predictor, zond, test):
    device = test[0].device
    gpus = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=gpus):        # sigreg uses torch.randn: the global dice stay as they were
        torch.manual_seed(0)
        good = sigreg(torch.randn(CLOUD, LATENT, device=device)).item()
        bad = sigreg(torch.zeros(CLOUD, LATENT, device=device)).item()
    assert good < 3, f"sigreg on a perfect cloud = {good:.2f}, should be about 1"
    assert bad > 50, f"sigreg on a collapsed cloud = {bad:.2f}, should be big (about 100)"


def check_no_nan(encoder, predictor, zond, test):
    for name, x in zip(["lidar_now", "action", "speed", "lidar_next"], test):
        assert torch.isfinite(x).all(), f"{name} has NaN or inf"


SMALL_CHECKS = [check_shapes, check_probe, check_sigreg, check_no_nan]


def check_code(encoder, predictor, zond, test):
    ok = True
    for small_check in SMALL_CHECKS:
        try:
            small_check(encoder, predictor, zond, test)
            print(f"OK    {small_check.__name__}")
        except AssertionError as error:
            print(f"FAIL  {small_check.__name__}: {error}")
            ok = False
    return {"code_ok": ok}


if __name__ == "__main__":
    test = load_pairs("test")
    encoder, predictor = load_jepa()
    zond = load_zond("zond.pt")
    print(check_code(encoder, predictor, zond, test))
