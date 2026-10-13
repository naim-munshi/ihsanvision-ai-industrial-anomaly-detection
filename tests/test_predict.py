import base64
from io import BytesIO

import numpy as np
from PIL import Image

from src.predict import _overlay, _first_scalar


def test_scalar_extraction():
    assert _first_scalar(np.asarray([0.75])) == 0.75
    assert _first_scalar(None) is None


def test_heatmap_is_valid_png():
    source = Image.new("RGB", (16, 12), "white")
    encoded = _overlay(source, np.arange(4, dtype=np.float32).reshape(2, 2))
    decoded = Image.open(BytesIO(base64.b64decode(encoded)))
    assert decoded.size == (16, 12)
    assert decoded.format == "PNG"


def test_empty_anomaly_map_is_optional():
    assert _overlay(Image.new("RGB", (10, 10)), None) is None
