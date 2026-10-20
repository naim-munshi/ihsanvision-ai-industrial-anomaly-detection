# Five-category real-sample acceptance (private photos, no training)

The owner generated a 10-image private archive from the **original MVTec AD test directories**: normal.png and defective.png for bottle, cable, carpet, metal_nut and capsule. These images may overlap the benchmark holdout; they are **manual demonstrations**, not an independent accuracy test. The new CLI checks real HTTP inference through each saved checkpoint, per-model/category routing, valid overlays and observed label decisions without reproducing any private dataset content.

## Step 1: download and privately unpack the Kaggle ZIP

Download the previously generated Kaggle Output file named ihsanvision-private-ui-samples.zip into the Mac Downloads directory. Do not upload these licensed images to public GitHub.

Run the following in a Mac Terminal (a brand-new private destination is required):

~~~bash
umask 077
python3 - <<'PY'
from pathlib import Path
from zipfile import ZipFile
import os
import shutil

source = Path.home() / "Downloads/ihsanvision-private-ui-samples.zip"
target = Path.home() / "ihsanvision-artifacts/private-ui-samples"

categories = ("bottle", "cable", "carpet", "metal_nut", "capsule")
expected = {
    f"{category}/{kind}.png"
    for category in categories for kind in ("normal", "defective")
}

assert source.is_file(), f"Missing Kaggle download: {source}"
assert not target.exists(), f"Refusing to overwrite: {target}"

with ZipFile(source) as archive:
    entries = archive.infolist()
    assert {entry.filename for entry in entries} == expected
    assert len(entries) == 10
    assert all(not entry.is_dir() and 0 < entry.file_size <= 10_000_000
               and ((entry.external_attr >> 16) & 0o170000) != 0o120000
               for entry in entries), "Invalid ZIP entries"
    assert archive.testzip() is None

    target.mkdir(mode=0o700)
    for entry in entries:
        dest = target / entry.filename
        dest.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with archive.open(entry) as stream, dest.open("xb") as output:
            shutil.copyfileobj(stream, output)
        os.chmod(dest, 0o600)

print("10 private demonstration PNGs:", target)
PY
~~~

## Step 2: test one category, then all 10 checkpoints

Keep the authenticated Mac localhost server running on port 8004. Run this in a second Terminal:

~~~bash
cd "$HOME/ihsanvision-ai-industrial-anomaly-detection"
source "$HOME/ihsanvision-artifacts/venv-py312/bin/activate"
git pull --ff-only origin main

python -m scripts.multicategory_real_sample_acceptance \
  --url http://127.0.0.1:8004 \
  --categories cable \
  --samples-dir "$HOME/ihsanvision-artifacts/private-ui-samples" \
  --key-file "$HOME/ihsanvision-artifacts/portfolio-operator-key.txt" \
  --timeout 300
~~~

If Cable works, repeat with all categories and an exclusive private evidence file:

~~~bash
python -m scripts.multicategory_real_sample_acceptance \
  --url http://127.0.0.1:8004 \
  --categories bottle cable carpet metal_nut capsule \
  --samples-dir "$HOME/ihsanvision-artifacts/private-ui-samples" \
  --key-file "$HOME/ihsanvision-artifacts/portfolio-operator-key.txt" \
  --timeout 300 \
  --output "$HOME/ihsanvision-artifacts/private-real-sample-acceptance.json"
~~~

There should be 20 actual HTTP requests (five categories times two models times normal/defective). API PASSED means routing, response contract and output PNG validity passed, NOT that defect diagnosis was correct. A threshold-label MISMATCH is recorded, not erased or mislabeled as success. The count is a descriptive tally of **manually selected examples**, never a recall, precision, sensitivity or accuracy estimate. Different model/category raw scores are not calibrated probabilities and must not be ranked against each other.

The CLI never prints keys, image data, overlay base64 or checkpoint paths. Its optional report contains only locally generated metadata and is owner-only 0600; files cannot be overwritten. When private history is on, the script deletes **only the newly returned inspection ID for its own request**. A lost HTTP response or abrupt interruption may leave test entries; review the private history manually instead of bulk deleting unrelated records.

## Step 3: manual operator browser still mandatory

Open http://127.0.0.1:8004/ on the Mac. Select an appropriate product category and PaDiM/PatchCore model, enter the private operator key in the local browser control, upload that category's normal.png and defective.png, and inspect the displayed heatmaps. Also test the /batch and /history screens, lot grouping, human verdict changes, CSV export and restart persistence. The CLI does **not** click the UI or certify heatmap localization. Do not share private dataset images, operator credentials or unapproved demonstration media.

Related: [five-category scope](portfolio-five-category-extension.md) and [photo-free smoke tests](five-category-synthetic-smoke.md).