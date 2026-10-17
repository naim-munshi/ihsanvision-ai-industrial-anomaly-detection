"""No-model-runtime tests of staged operator approval and rollback safety."""
import hashlib
import json
from pathlib import Path

import pytest

from scripts.stage_model_release import stage_release, verify_release, main


def candidates(tmp_path: Path):
    weights = tmp_path / "weights"
    weights.mkdir()
    models = []
    checkpoints = {}
    for name in ("padim", "patchcore"):
        path = weights / f"{name}.ckpt"
        path.write_bytes(name.encode("ascii"))
        checkpoints[name] = path
        models.append({
            "category": "bottle", "model": name,
            "checkpoint": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "seed": 42, "protocol": "disjoint_heldout_test",
        })
    source = tmp_path / "candidates.json"
    source.write_text(json.dumps({
        "version": 1, "status": "candidates_only_not_approved",
        "models": models,
    }))
    return source, checkpoints


def test_staged_bundle_validates_for_existing_apis(tmp_path, monkeypatch):
    source, _ = candidates(tmp_path)
    folder = tmp_path / "v1"
    stage_release(source, folder, ["bottle:padim", "bottle:patchcore"], reviewed=True)
    assert verify_release(folder) == 2
    assert (folder / "model-registry.json").stat().st_mode & 0o077 == 0
    assert (folder / "approved-models.json").stat().st_mode & 0o077 == 0
    monkeypatch.setenv("IHSANVISION_MODEL_REGISTRY", str(folder / "model-registry.json"))
    monkeypatch.setenv("IHSANVISION_APPROVED_MODELS", str(folder / "approved-models.json"))
    from ihsanvision.registry import configured_registry
    from ihsanvision.approvals import configured_approvals
    registered = configured_registry()
    pins = configured_approvals()
    assert set(registered["bottle"]) == {"padim", "patchcore"}
    assert set(pins) == {("bottle", "padim"), ("bottle", "patchcore")}


def test_no_automatic_approval_without_operator_review(tmp_path):
    source, _ = candidates(tmp_path)
    folder = tmp_path / "release"
    with pytest.raises(ValueError, match="Manual"):
        stage_release(source, folder, ["bottle:padim"], reviewed=False)
    assert not folder.exists()


def test_refuse_modified_checkpoint_before_staging(tmp_path):
    source, checkpoints = candidates(tmp_path)
    checkpoints["padim"].write_bytes(b"mutated")
    output = tmp_path / "release"
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        stage_release(source, output, ["bottle:padim"], reviewed=True)
    assert not output.exists()


def test_verify_blocks_mutation_before_activation(tmp_path):
    source, checkpoints = candidates(tmp_path)
    folder = tmp_path / "v1"
    stage_release(source, folder, ["bottle:padim"], reviewed=True)
    checkpoints["padim"].write_bytes(b"tampered")
    with pytest.raises(ValueError, match="integrity mismatch"):
        verify_release(folder)


def test_release_creation_is_exclusive(tmp_path):
    source, _ = candidates(tmp_path)
    folder = tmp_path / "v1"
    stage_release(source, folder, ["bottle:padim"], reviewed=True)
    with pytest.raises(ValueError, match="new absolute directory"):
        stage_release(source, folder, ["bottle:patchcore"], reviewed=True)
    assert verify_release(folder) == 1


def test_duplicate_and_unknown_pairs_are_refused(tmp_path):
    source, _ = candidates(tmp_path)
    with pytest.raises(ValueError, match="distinct"):
        stage_release(source, tmp_path / "v1",
                      ["bottle:padim", "bottle:padim"], reviewed=True)
    with pytest.raises(ValueError, match="absent"):
        stage_release(source, tmp_path / "v2",
                      ["cable:padim"], reviewed=True)


def test_symlinked_checkpoint_is_refused(tmp_path):
    source, checkpoints = candidates(tmp_path)
    data = json.loads(source.read_text())
    link = tmp_path / "linked.ckpt"
    link.symlink_to(checkpoints["padim"])
    data["models"][0]["checkpoint"] = str(link)
    source.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="regular"):
        stage_release(source, tmp_path / "v1", ["bottle:padim"], reviewed=True)


def test_corrupt_approval_blocks_release_verification(tmp_path):
    source, _ = candidates(tmp_path)
    folder = tmp_path / "v1"
    stage_release(source, folder, ["bottle:padim"], reviewed=True)
    approval = folder / "approved-models.json"
    value = json.loads(approval.read_text())
    value["approvals"][0]["checkpoint_sha256"] = "f" * 64
    approval.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="integrity mismatch"):
        verify_release(folder)


def test_cli_defaults_do_not_approve(tmp_path):
    source, _ = candidates(tmp_path)
    with pytest.raises(SystemExit, match="Manual"):
        main(["stage", "--candidates", str(source), "--output-dir",
              str(tmp_path / "v1"), "--approve", "bottle:padim"])
    assert not (tmp_path / "v1").exists()
