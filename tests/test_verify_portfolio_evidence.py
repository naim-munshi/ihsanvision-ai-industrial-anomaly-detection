"""Dependency-isolated and checkpoint-free portfolio evidence verifier tests."""
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

from PIL import Image
import pytest

from scripts import verify_portfolio_evidence as gate


def artifacts(tmp_path):
    root = tmp_path / "private-run"
    root.mkdir()
    lines = []
    for name in sorted(gate.REQUIRED):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        data = ("fixture:" + name).encode()
        path.write_bytes(data)
        lines.append(f"{sha256(data).hexdigest()}  {name}")
    (root / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "benchmark_summary.md").write_text("reproduced-summary\n", encoding="utf-8")
    smoke = tmp_path / "smoke.zip"
    png = BytesIO()
    Image.new("RGB", (5, 6), "green").save(png, "PNG")
    with ZipFile(smoke, "x") as archive:
        archive.writestr("smoke_report.json", json.dumps({
            "model": "patchcore",
            "status": "single-image-inference-passed",
            "predicted_anomaly": True,
        }))
        archive.writestr("smoke_overlay.png", png.getvalue())
    return root, smoke


def test_all_private_evidence_checked_without_loading_weights(monkeypatch, tmp_path):
    root, smoke = artifacts(tmp_path)
    calls = []
    def fake_summary(path):
        calls.append(Path(path))
        return "reproduced-summary\n"
    monkeypatch.setattr(gate, "summarize", fake_summary)
    result = gate.verify(root, smoke)
    assert result["status"] == "artifact_evidence_verified"
    assert result["benchmark_manifest"]["files_verified"] == 8
    assert result["benchmark"]["saved_summary_matches"] is True
    assert result["patchcore_smoke"]["overlay_dimensions"] == [5, 6]
    assert result["public_release_authorized"] is False
    assert result["scientific_replication"] is False
    assert calls == [root]
    dump = json.dumps(result)
    assert str(tmp_path) not in dump and "fixture:padim" not in dump


def test_changed_checkpoint_fails_checksum(monkeypatch, tmp_path):
    root, smoke = artifacts(tmp_path)
    (root / "patchcore-bottle-seed42" / "model.ckpt").write_bytes(b"modified")
    monkeypatch.setattr(gate, "summarize", lambda _: "reproduced-summary\n")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        gate.verify(root, smoke)


def test_manifest_extra_path_and_duplicates_refused(tmp_path):
    root, _ = artifacts(tmp_path)
    manifest = root / "SHA256SUMS.txt"
    original = manifest.read_text()
    manifest.write_text(original + ("a" * 64) + "  ../secret\n")
    with pytest.raises(ValueError, match="Unexpected/duplicate"):
        gate.verify_manifest(root)
    manifest.write_text(original + original.splitlines()[0] + "\n")
    with pytest.raises(ValueError, match="Unexpected/duplicate"):
        gate.verify_manifest(root)


def test_missing_summary_and_mismatch(monkeypatch, tmp_path):
    root, _ = artifacts(tmp_path)
    monkeypatch.setattr(gate, "summarize", lambda _: "different")
    with pytest.raises(ValueError, match="differs"):
        gate.verify_summary(root)
    (root / "benchmark_summary.md").unlink()
    with pytest.raises(FileNotFoundError):
        gate.verify_summary(root)


def test_invalid_smoke_report_or_overlay_refused(tmp_path):
    _, smoke = artifacts(tmp_path)
    with ZipFile(smoke, "r") as archive:
        original_png = archive.read("smoke_overlay.png")
    smoke.unlink()
    with ZipFile(smoke, "x") as archive:
        archive.writestr("smoke_report.json", json.dumps({
            "model": "patchcore", "status": "single-image-inference-passed",
            "predicted_anomaly": False,
        }))
        archive.writestr("smoke_overlay.png", original_png)
    with pytest.raises(ValueError, match="Expected observed"):
        gate.verify_smoke_zip(smoke)
    smoke.unlink()
    with ZipFile(smoke, "x") as archive:
        archive.writestr("smoke_report.json", json.dumps({
            "model": "patchcore", "status": "single-image-inference-passed",
            "predicted_anomaly": True,
        }))
        archive.writestr("smoke_overlay.png", b"not-a-png")
    with pytest.raises(OSError):
        gate.verify_smoke_zip(smoke)


def test_cli_never_overwrites_prior_record(tmp_path):
    path = tmp_path / "audit.json"
    path.write_text('{"original":true}')
    with pytest.raises(SystemExit) as exc:
        gate.main(["--runs-dir", str(tmp_path), "--smoke-zip", str(tmp_path / "x.zip"),
                   "--output", str(path)])
    assert exc.value.code == 2
    assert json.loads(path.read_text()) == {"original": True}
