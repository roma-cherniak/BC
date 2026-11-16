"""The loop in a temporary git repo, with the dummy agent and a fake judge: no AIL Cup, no Claude."""
import csv
import random
import re
import subprocess
from types import SimpleNamespace

import pytest

from autoresearch.loop import LoopStopped, dummy_agent, git, run_loop
from autoresearch.score import NotScorable, Score

CONTROLLER = "SIDE_MARGIN = 0.45\nSTART_SPEED = 0.8\nBUDGET_SHARE = 0.3\n"


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "ail_cup").mkdir()
    (tmp_path / "ail_cup" / "simple_controller.py").write_text(CONTROLLER)
    (tmp_path / "autoresearch").mkdir()
    (tmp_path / "autoresearch" / "program.md").write_text("improve the car")
    for args in (["init", "-q", "-b", "main"], ["config", "user.email", "t@t"], ["config", "user.name", "t"],
                 ["add", "."], ["commit", "-qm", "start"], ["branch", "autoresearch/base"]):
        subprocess.run(["git", *args], cwd=tmp_path, check=True)
    return tmp_path


def fake_judge(scores):
    """Returns the next score of the list for each judged experiment."""
    it = iter(scores)

    def judge_fn(controller, label):
        s = next(it)
        if s is None:
            raise NotScorable("platform down")
        return SimpleNamespace(score=Score(s, {"fiit-oval": s / 3, "hairpin-valley": s / 3, "danube-chicane": s / 3}, ()))
    return judge_fn


def rows(repo, run=1):
    with (repo / "autoresearch" / "runs" / f"run{run}" / "results.tsv").open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def agents():
    rng = random.Random(0)
    return {"dummy": lambda p, root: dummy_agent(p, root, rng)}


def test_better_is_committed_worse_is_undone(repo):
    run_loop(1, "dummy", 3, fake_judge([330.0, 340.0, 329.0]), root=repo, agents=agents())
    r = rows(repo)
    assert [x["kept"] for x in r] == ["yes", "yes", "no", "no"]   # 329 is only 1 s better than 330: noise
    assert float(r[-1]["best_s"]) == 330.0
    assert git(repo, "branch", "--show-current") == "autoresearch/run1"
    assert git(repo, "log", "--oneline").count("autoresearch exp") == 1
    assert git(repo, "status", "--porcelain", "--", "ail_cup") == ""      # the losing change was undone


def test_the_kept_file_is_the_one_in_the_commit(repo):
    run_loop(1, "dummy", 1, fake_judge([300.0]), root=repo, agents=agents())
    committed = git(repo, "show", "HEAD:ail_cup/simple_controller.py")
    assert committed != CONTROLLER.strip() and re.search(r"^SIDE_MARGIN|^START_SPEED|^BUDGET_SHARE", committed, re.M)


def test_a_forbidden_file_stops_the_loop(repo):
    def cheater(prompt, root):
        (root / "autoresearch" / "score.py").write_text("cheat")
        return "CHANGE: edited the judge"
    with pytest.raises(LoopStopped, match="forbidden"):
        run_loop(1, "cheat", 1, fake_judge([1.0]), root=repo, agents={"cheat": cheater})


def test_not_scorable_is_logged_and_undone(repo):
    run_loop(1, "dummy", 1, fake_judge([None]), root=repo, agents=agents())
    assert rows(repo)[-1]["dnf"].startswith("not scored")
    assert git(repo, "status", "--porcelain", "--", "ail_cup") == ""


def test_every_run_starts_from_the_base_not_from_the_last_run(repo):
    base = git(repo, "rev-parse", "autoresearch/base")
    run_loop(1, "dummy", 1, fake_judge([300.0]), root=repo, agents=agents())     # run 1 commits a change
    run_loop(2, "dummy", 0, fake_judge([]), root=repo, agents=agents())          # run 2 is created now
    assert git(repo, "rev-parse", "autoresearch/run1") != base
    assert git(repo, "rev-parse", "autoresearch/run2") == base
    assert git(repo, "rev-parse", "main") == base                                # main never moves


def test_a_missing_base_branch_stops_the_loop(repo):
    git(repo, "branch", "-D", "autoresearch/base")
    with pytest.raises(LoopStopped, match="base branch"):
        run_loop(1, "dummy", 1, fake_judge([1.0]), root=repo, agents=agents())


def test_a_stopped_run_continues_with_its_best(repo):
    run_loop(1, "dummy", 1, fake_judge([330.0]), root=repo, agents=agents())
    run_loop(1, "dummy", 1, fake_judge([331.0]), root=repo, agents=agents())
    r = rows(repo)
    assert [x["exp"] for x in r] == ["0", "1", "2"] and float(r[-1]["best_s"]) == 330.0
