import json

import pytest

from emergent_garden.config import Config
from emergent_garden.history import read_history


def segment(path, times, population=2, parent=None):
    path.mkdir()
    (path / "config.toml").write_text("ecology_version = 13\n")
    (path / "metadata.json").write_text(
        json.dumps(dict(seed=1, ablation="none", resumed_from=str(parent) if parent else None))
    )
    (path / "metrics.jsonl").write_text(
        "\n".join(
            json.dumps(dict(tick=t, time=t, population=population, speed=t - times[0]))
            for t in times
        )
    )
    (path / "events.jsonl").write_text(json.dumps(dict(event="birth", time=times[-1])))


def test_complete_resumed_history_keeps_events_once_and_checks_physical_boundary(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    segment(first, [0, 30])
    segment(second, [30, 60], parent=first / "latest.pt")
    pieces, rows, events = read_history(second, follow_resumes=True)
    assert [s["path"] for s in pieces] == [first, second]
    assert rows[0]["tick"] == 0 and rows[-1]["tick"] == 60
    assert [e["time"] for e in events] == [30, 60]
    assert len(read_history(second)[0]) == 1


def test_resumed_history_accepts_explicit_neutral_defaults_from_newer_releases(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    segment(first, [0, 30])
    segment(second, [30, 60], parent=first / "latest.pt")
    c = Config.load(second / "config.toml")
    c.save(second / "config.toml")
    assert (first / "config.toml").read_bytes() != (second / "config.toml").read_bytes()
    assert len(read_history(second, follow_resumes=True)[0]) == 2
    c.handling_rate *= 2
    c.save(second / "config.toml")
    with pytest.raises(ValueError, match="changed configuration"):
        read_history(second, follow_resumes=True)


@pytest.mark.parametrize("problem", ["population", "configuration", "seed", "time"])
def test_changed_resume_is_not_reported_as_one_continuous_experiment(tmp_path, problem):
    first, second = tmp_path / "first", tmp_path / "second"
    segment(first, [0, 30])
    segment(
        second,
        [31 if problem == "time" else 30, 60],
        population=3 if problem == "population" else 2,
        parent=first / "latest.pt",
    )
    if problem == "configuration":
        (second / "config.toml").write_text("ecology_version = 14\n")
    if problem == "seed":
        path = second / "metadata.json"
        data = json.loads(path.read_text())
        path.write_text(json.dumps(dict(data, seed=2)))
    with pytest.raises(ValueError, match="boundary|changed"):
        read_history(second, follow_resumes=True)


def test_missing_ancestry_and_resume_cycles_fail_explicitly(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    segment(first, [20, 30])
    with pytest.raises(ValueError, match="Missing initialization"):
        read_history(first, follow_resumes=True)
    segment(second, [30, 60], parent=first / "latest.pt")
    path = first / "metadata.json"
    data = json.loads(path.read_text())
    path.write_text(json.dumps(dict(data, resumed_from=str(second / "latest.pt"))))
    with pytest.raises(ValueError, match="cycle"):
        read_history(second, follow_resumes=True)
