# Imagination: one step of the predictor, with or without guard B.

from jepa.jepa_model import LATENT


def imagine_step(predictor, z, act, spd, guard):

    z = predictor(z, act, spd)
    if guard:
        z = z / z.norm(dim=1, keepdim=True) * LATENT ** 0.5
    return z
