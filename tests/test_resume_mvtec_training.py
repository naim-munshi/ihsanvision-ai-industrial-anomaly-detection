"""Dry-run and bounded resumability contracts; no ML datasets or weights required."""
import json

import pytest

from ihsanvision.config import TrainConfig
from scripts import resume_mvtec_training as runner


def local_dataset(tmp_path, categories=("bottle",)):
    root = tmp_path / "dataset"
    for category in categories:
        for relative in ("train/good/1.png", "test/good/2.png",
                         "test/broken/3.png"):
            path = root / category / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"test-file-stub")
    return root


def fake_artifacts(runs, category, model, seed, *, train_paths=None):
    folder = runs / f"{model}-{category}-seed{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": train_paths or ["/dataset/train/good/1.png"],
        "val": ["/dataset/test/broken/3.png"],
        "test": ["/dataset/test/good/2.png"],
    }
    (folder / "model.ckpt").write_bytes(b"fake-not-a-real-lightning-checkpoint")
    (folder / "splits.json").write_text(json.dumps(paths))
    (folder / "report.json").write_text(json.dumps({
        "model": model, "category": category,
        "config": {"seed": seed},
        "split_counts": {k: len(v) for k, v in paths.items()},
        "train_seconds": 1.75,
        "evaluation": {"metrics": {
            "image_AUROC": 0.9, "image_F1Score": 0.8,
            "pixel_AUROC": 0.8, "pixel_F1Score": 0.7,
        }},
    }))
    return folder


def test_all_missing_jobs_are_planned_without_writing(tmp_path):
    dataset = local_dataset(tmp_path, ("bottle", "cable"))
    runs = tmp_path / "output"
    result = runner.plan(dataset, runs, ["bottle", "cable"], 42)
    assert result["status"] == "jobs_missing"
    assert result["jobs"] == [
        {"category": "bottle", "model": "padim"},
        {"category": "bottle", "model": "patchcore"},
        {"category": "cable", "model": "padim"},
        {"category": "cable", "model": "patchcore"},
    ]
    assert result["dataset_counts"][0]["train_good"] == 1
    assert str(tmp_path) not in json.dumps(result)
    assert not runs.exists()


def test_completed_runs_reused_and_matching_pair_verified(tmp_path):
    dataset = local_dataset(tmp_path)
    runs = tmp_path / "runs"
    fake_artifacts(runs, "bottle", "padim", 42)
    result = runner.plan(dataset, runs, ["bottle"], 42)
    assert result["jobs"] == [{"category": "bottle", "model": "patchcore"}]
    fake_artifacts(runs, "bottle", "patchcore", 42)
    finished = runner.plan(dataset, runs, ["bottle"], 42)
    assert finished["status"] == "all_completed"
    assert finished["jobs"] == []


def test_split_mismatch_fails_before_any_new_experiment(tmp_path):
    dataset = local_dataset(tmp_path)
    runs = tmp_path / "runs"
    fake_artifacts(runs, "bottle", "padim", 42)
    fake_artifacts(runs, "bottle", "patchcore", 42,
                   train_paths=["/other/folder/image.png"])
    with pytest.raises(RuntimeError, match="Unfair benchmark"):
        runner.plan(dataset, runs, ["bottle"], 42)


@pytest.mark.parametrize("partial", ["splits.json", "report.json", "model.ckpt"])
def test_partial_artifacts_block_resumption(tmp_path, partial):
    dataset = local_dataset(tmp_path)
    runs = tmp_path / "runs"
    path = runs / "padim-bottle-seed42" / partial
    path.parent.mkdir(parents=True)
    path.write_bytes(b"unverified-data")
    with pytest.raises(RuntimeError, match="Partial"):
        runner.plan(dataset, runs, ["bottle"], 42)
    assert path.read_bytes() == b"unverified-data"


def test_validates_all_categories_before_any_fit(tmp_path):
    dataset = local_dataset(tmp_path, ("bottle",))
    with pytest.raises(ValueError, match="Missing MVTec"):
        runner.plan(dataset, tmp_path / "runs", ["bottle", "cable"], 42)
    assert not (tmp_path / "runs").exists()


def test_resume_one_job_at_a_time_and_enforce_shared_splits(tmp_path):
    dataset = local_dataset(tmp_path)
    runs = tmp_path / "runs"
    seen: list[tuple[str, str | None]] = []

    def trainer(config: TrainConfig):
        seen.append((
            config.model,
            None if config.reference_splits is None else
            config.reference_splits.name,
        ))
        fake_artifacts(runs, config.category, config.model, config.seed)
        return {"status": "test_stub"}

    first = runner.plan(dataset, runs, ["bottle"], 42)
    assert runner.execute(first, dataset, runs, seed=42, accelerator="cpu",
                          max_jobs=1, trainer=trainer) == [
        {"category": "bottle", "model": "padim", "status": "trained_and_verified"}
    ]
    second = runner.plan(dataset, runs, ["bottle"], 42)
    assert second["jobs"] == [{"category": "bottle", "model": "patchcore"}]
    runner.execute(second, dataset, runs, seed=42, accelerator="cpu",
                   max_jobs=1, trainer=trainer)
    assert seen == [("padim", None), ("patchcore", "splits.json")]
    assert runner.plan(dataset, runs, ["bottle"], 42)["status"] == "all_completed"


def test_existing_patchcore_sets_reference_for_missing_padim(tmp_path):
    dataset = local_dataset(tmp_path)
    runs = tmp_path / "runs"
    fake_artifacts(runs, "bottle", "patchcore", 42)
    state = runner.plan(dataset, runs, ["bottle"], 42)
    captured = []

    def trainer(config):
        captured.append(config.reference_splits)
        fake_artifacts(runs, config.category, config.model, config.seed)

    runner.execute(state, dataset, runs, seed=42, accelerator="cpu",
                   trainer=trainer)
    assert captured == [runs / "patchcore-bottle-seed42" / "splits.json"]


def test_execute_refuses_race_and_cannot_overwrite(tmp_path):
    dataset = local_dataset(tmp_path)
    runs = tmp_path / "runs"
    state = runner.plan(dataset, runs, ["bottle"], 42)
    fake_artifacts(runs, "bottle", "padim", 42)
    with pytest.raises(RuntimeError, match="appeared since planning"):
        runner.execute(state, dataset, runs, seed=42, accelerator="cpu",
                       trainer=lambda _: pytest.fail("Must not train"))


def test_bounded_max_jobs_and_invalid_input(tmp_path):
    dataset = local_dataset(tmp_path)
    state = runner.plan(dataset, tmp_path / "runs", ["bottle"], 42)
    with pytest.raises(ValueError, match="max_jobs"):
        runner.execute(state, dataset, tmp_path / "runs", seed=42,
                       accelerator="cpu", max_jobs=5)
    with pytest.raises(ValueError, match="nonduplicate"):
        runner.plan(dataset, tmp_path / "runs", ["bottle", "bottle"], 42)
    with pytest.raises(ValueError, match="Seed"):
        runner.plan(dataset, tmp_path / "runs", ["bottle"], -1)


def test_cli_dry_run_does_not_fit_or_create_artifacts(tmp_path, monkeypatch, capsys):
    dataset = local_dataset(tmp_path)
    runs = tmp_path / "runs"
    monkeypatch.setattr(runner, "run", lambda *_: pytest.fail("Must not train"))
    runner.main(["--dataset-root", str(dataset), "--runs-dir", str(runs),
                 "--categories", "bottle"])
    assert '"training_jobs_pending": 2' in capsys.readouterr().out
    assert not runs.exists()
