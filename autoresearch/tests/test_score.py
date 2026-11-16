import pytest

from autoresearch.score import NotScorable, is_better, score


def row(time_s=None, finished=True, sim_error=None):
    return {"run_id": 1, "finished": finished, "total_time_s": time_s, "sim_error": sim_error}


BASELINE = {"fiit-oval": row(98.05), "hairpin-valley": row(125.01), "danube-chicane": row(114.75)}


def test_sum_of_finished_tracks_is_the_baseline():
    s = score(BASELINE)
    assert s.total_s == pytest.approx(337.81)
    assert s.dnf_tracks == ()


def test_a_dnf_counts_as_the_penalty_and_never_wins():
    fast_but_crashes = {"fiit-oval": row(85.0), "hairpin-valley": row(finished=False), "danube-chicane": row(100.0)}
    s = score(fast_but_crashes)
    assert s.per_track_s["hairpin-valley"] == 300.0
    assert s.dnf_tracks == ("hairpin-valley",)
    assert not is_better(s.total_s, 337.81)


def test_a_missing_track_cannot_be_scored():
    with pytest.raises(NotScorable, match="danube-chicane"):
        score({"fiit-oval": row(98.0), "hairpin-valley": row(125.0)})


def test_a_platform_error_cannot_be_scored():
    rows = dict(BASELINE, **{"fiit-oval": row(sim_error="sim crashed")})
    with pytest.raises(NotScorable, match="platform"):
        score(rows)


@pytest.mark.parametrize("new_s, kept", [(336.0, True), (336.4, False), (337.81, False), (340.0, False)])
def test_keep_only_beyond_the_noise(new_s, kept):
    assert is_better(new_s, 337.81) is kept
