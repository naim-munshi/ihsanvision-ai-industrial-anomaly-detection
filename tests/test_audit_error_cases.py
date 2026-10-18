"""Do not create portfolio error claims without real paired test prediction rows."""
import csv
from io import StringIO

import pytest

from scripts.audit_error_cases import COLUMNS, main, read_cases, summarize_cases


def cases_file(tmp_path, *, categories=("bottle", "cable", "carpet")):
    path = tmp_path / "private.csv"
    rows = []
    for category in categories:
        for model in ("padim", "patchcore"):
            for identifier, label, predicted, score in (
                (f"{category}_n_0001", "normal", "false", "0.12"),
                (f"{category}_n_0002", "normal", "true", "0.76"),
                (f"{category}_a_0001", "anomaly", "true", "0.9"),
                (f"{category}_a_0002", "anomaly", "false", "0.2"),
            ):
                rows.append([category, model, identifier, "test", label, predicted, score])
    write_cases(path, rows)
    return path


def write_cases(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(COLUMNS)
        writer.writerows(rows)


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        return list(reader)


def test_real_paired_cases_yield_confusion_counts_and_bounded_examples(tmp_path):
    source = cases_file(tmp_path)
    observed = read_cases(source)
    report = summarize_cases(observed)
    for category in ("bottle", "cable", "carpet"):
        assert f"| {category} | padim | 4 | 1 | 1 | 1 | 1 | 0.500 | 0.500 | 0.500 |" in report
        assert f"| {category} | patchcore | 4 | 1 | 1 | 1 | 1 | 0.500 | 0.500 | 0.500 |" in report
    assert "false positive(s)" in report
    assert "none were observed" in report
    assert "/tmp/" not in report


def test_cli_refuses_to_overwrite_previous_private_analysis(tmp_path):
    source = cases_file(tmp_path)
    out = tmp_path / "analysis.md"
    main(["--input", str(source), "--output", str(out)])
    assert "Portfolio test-set classification error audit" in out.read_text()
    assert out.stat().st_mode & 0o077 == 0
    with pytest.raises(FileExistsError):
        main(["--input", str(source), "--output", str(out)])


@pytest.mark.parametrize("modification,match", [
    (lambda rows: rows.pop(), "Unfair sample IDs"),
    (lambda rows: rows.append(rows[0][:]), "Duplicate image/model"),
    (lambda rows: rows.__setitem__(0, [*rows[0][:5], "not-a-boolean", rows[0][-1]]),
     "predicted_anomaly"),
    (lambda rows: rows.__setitem__(0, [*rows[0][:6], "nan"]), "finite"),
    (lambda rows: rows.__setitem__(0, [*rows[0][:6], "inf"]), "finite"),
    (lambda rows: rows[0].__setitem__(3, "train"), "held-out test"),
    (lambda rows: rows[0].__setitem__(2, "../sensitive.png"), "opaque"),
    (lambda rows: rows[0].__setitem__(4, "maybe"), "ground_truth"),
    (lambda rows: rows[0].__setitem__(0, "hazelnut"), "portfolio category"),
])
def test_bad_input_is_rejected(tmp_path, modification, match):
    source = cases_file(tmp_path)
    rows = read_rows(source)
    modification(rows)
    write_cases(source, rows)
    with pytest.raises(ValueError, match=match):
        read_cases(source)


def test_disagreement_on_same_ground_truth_is_rejected(tmp_path):
    source = cases_file(tmp_path)
    rows = read_rows(source)
    for row in rows:
        if row[0] == "cable" and row[1] == "patchcore" and row[2] == "cable_n_0001":
            row[4] = "anomaly"
    write_cases(source, rows)
    with pytest.raises(ValueError, match="ground-truth labels"):
        read_cases(source)


def test_file_with_only_normal_images_is_rejected(tmp_path):
    source = cases_file(tmp_path)
    rows = read_rows(source)
    for row in rows:
        if row[0] == "carpet":
            row[4] = "normal"
    write_cases(source, rows)
    with pytest.raises(ValueError, match="both normal and anomalous"):
        read_cases(source)


def test_reject_absent_models_and_relative_or_symlink_files(tmp_path):
    source = cases_file(tmp_path, categories=("bottle",))
    with pytest.raises(ValueError, match="Both paired models"):
        read_cases(source, categories=("bottle", "cable"))
    with pytest.raises(ValueError, match="absolute private"):
        read_cases(source.name, categories=("bottle",))
    link = tmp_path / "linked.csv"
    link.symlink_to(source)
    with pytest.raises(ValueError, match="absolute private"):
        read_cases(link, categories=("bottle",))


def test_missing_metrics_are_not_assumed(tmp_path):
    source = cases_file(tmp_path)
    rows = read_rows(source)
    rows[0][6] = ""
    write_cases(source, rows)
    with pytest.raises(ValueError, match="numeric"):
        read_cases(source)


def test_missing_csv_columns_rejected(tmp_path):
    source = cases_file(tmp_path)
    text = source.read_text()
    source.write_text(text.replace("anomaly_score", "confidence", 1))
    with pytest.raises(ValueError, match="schema"):
        read_cases(source)
