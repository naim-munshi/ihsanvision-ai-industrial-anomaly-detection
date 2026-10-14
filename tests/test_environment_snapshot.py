"""Snapshot capture needs neither datasets nor Anomalib model initialization."""
import json

import pytest

from scripts import environment_snapshot as snapshot


def test_collect_records_versions_device_and_non_destructive_artifact_presence(tmp_path, monkeypatch):
    monkeypatch.setattr(snapshot, "package_versions", lambda: {"numpy": "2.0.2"})
    monkeypatch.setattr(snapshot, "accelerator_info", lambda: {"cuda_available": False})
    monkeypatch.setattr(snapshot, "git_value", lambda root, *args: "abc123" if args[-1] == "HEAD" else "")
    runs = tmp_path / "runs"
    (runs / "padim-bottle-seed42").mkdir(parents=True)
    (runs / "padim-bottle-seed42" / "model.ckpt").write_bytes(b"checkpoint-not-loaded")
    (runs / "padim-bottle-seed42" / "report.json").write_text("{}")
    record = snapshot.collect(tmp_path, runs)
    assert record["packages"] == {"numpy": "2.0.2"}
    assert record["accelerator"] == {"cuda_available": False}
    assert record["git"]["commit"] == "abc123"
    assert record["runs"]["padim-bottle-seed42"]["checkpoint_present"] is True
    assert record["runs"]["padim-bottle-seed42"]["splits_present"] is False
    assert record["runs"]["patchcore-bottle-seed42"]["checkpoint_present"] is False
    assert (runs / "padim-bottle-seed42/model.ckpt").read_bytes() == b"checkpoint-not-loaded"


def test_writes_json_only_to_new_file(tmp_path, monkeypatch):
    monkeypatch.setattr(snapshot, "collect", lambda *args: {"safe": True})
    target = tmp_path / "environment.json"
    snapshot.main(["--repo-dir", str(tmp_path), "--output", str(target)])
    assert json.loads(target.read_text()) == {"safe": True}
    with pytest.raises(FileExistsError):
        snapshot.main(["--repo-dir", str(tmp_path), "--output", str(target)])
    assert json.loads(target.read_text()) == {"safe": True}


def test_git_errors_are_nonfatal(tmp_path):
    assert snapshot.git_value(tmp_path, "rev-parse", "HEAD") is None


def test_no_output_mode_is_read_only(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(snapshot, "collect", lambda *args: {"ok": True})
    snapshot.main(["--repo-dir", str(tmp_path)])
    assert json.loads(capsys.readouterr().out) == {"ok": True}
    assert list(tmp_path.iterdir()) == []
