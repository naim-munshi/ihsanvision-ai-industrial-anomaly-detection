"""Repository license visibility and third-party data boundaries.

This is a documentation consistency test, NOT a legal enforceability audit.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_responsible_use_license_is_present_and_use_restricted():
    license_path = ROOT / "LICENSE"
    text = license_path.read_text(encoding="utf-8")
    assert text.startswith("IhsanVision.AI Responsible Use License")
    assert "Version 1.0" in text
    assert "NOT AN OSI-APPROVED OPEN-SOURCE LICENSE" in text
    assert "PERMISSION TO USE, STUDY, MODIFY AND SHARE" in text
    assert "PROHIBITED HARMFUL USE" in text
    assert "Commercial" in text or "commercial" in text
    for prohibited in (
        "Violence, weapons", "Deceptive or criminal conduct",
        "Privacy and surveillance abuse", "Discrimination and exploitation",
        "Dangerous industrial misuse", "safety-critical",
    ):
        assert prohibited in text
    assert "TERMINATION" in text
    assert "NO WARRANTY" in text


def test_code_permission_is_not_confused_with_mvtec_dataset_license():
    lic = (ROOT / "LICENSE").read_text(encoding="utf-8")
    guide = (ROOT / "docs/licensing.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "MVTec AD" in lic
    assert "CC BY-NC-SA 4.0" in lic
    assert "does NOT" in lic or "does not" in lic.lower()
    assert "commercial use of the dataset is not allowed" in guide
    assert "[IhsanVision.AI Responsible Use License v1.0](LICENSE)" in readme
    assert "not an OSI-approved open-source license" in readme
    assert "[licensing and permitted uses](docs/licensing.md)" in readme


def test_license_is_not_mistaken_for_full_portfolio_release():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    finishing = (ROOT / "FINISHING.md").read_text(encoding="utf-8")
    assert "No public hosted demo or v1.0 release is claimed today" in readme
    assert "Owner reviews the existing responsible-use source license" in finishing
    assert "dataset" in finishing.lower()
    assert not (ROOT / "docs/media/real-bottle-inspection.png").exists()
