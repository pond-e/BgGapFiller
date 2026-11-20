import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from main import InvalidImageError, app, fill_gap, most_used_color

client = TestClient(app)


def encode_png(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def make_image(height: int, width: int, bgr=(10, 20, 30)) -> np.ndarray:
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = bgr
    return img


def decode(jpeg: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)


def test_most_used_color_returns_mode():
    img = make_image(10, 10, (1, 2, 3))
    img[0, :3] = (200, 100, 50)
    assert most_used_color(img) == (1, 2, 3)


def test_wide_image_is_padded_vertically():
    # 横長 200x100 を縦長端末 (1:2) に合わせると高さ 400 になる
    out = decode(fill_gap(encode_png(make_image(100, 200)), 1, 2))
    assert out.shape[:2] == (400, 200)


def test_tall_image_is_padded_horizontally():
    out = decode(fill_gap(encode_png(make_image(400, 100)), 1, 2))
    assert out.shape[:2] == (400, 200)


def test_original_is_centered_and_padding_uses_mode_color():
    img = make_image(100, 100, (0, 0, 255))
    img[:60] = (255, 0, 0)  # 最頻色は青
    out = decode(fill_gap(encode_png(img), 1, 2))
    assert out.shape[:2] == (200, 100)
    # 上下の余白は最頻色（JPEG なので誤差を許容）
    assert np.allclose(out[5, 50], (255, 0, 0), atol=10)
    assert np.allclose(out[195, 50], (255, 0, 0), atol=10)
    # 元画像は中央（上から 50px〜150px）に置かれる
    assert np.allclose(out[140, 50], (0, 0, 255), atol=10)


@pytest.mark.parametrize(
    "img",
    [
        np.full((50, 50), 128, dtype=np.uint8),        # グレースケール
        np.full((50, 50, 4), 128, dtype=np.uint8),     # 透過 PNG
        np.full((50, 50, 3), 30000, dtype=np.uint16),  # 16bit PNG
    ],
)
def test_non_bgr_images_are_accepted(img):
    out = decode(fill_gap(encode_png(img), 1, 2))
    assert out.shape[:2] == (100, 50)


def test_invalid_inputs_raise():
    with pytest.raises(InvalidImageError):
        fill_gap(b"not an image", 1, 2)
    with pytest.raises(InvalidImageError):
        fill_gap(encode_png(make_image(10, 10)), 0, 2)
    with pytest.raises(InvalidImageError):
        fill_gap(encode_png(make_image(1000, 1000)), 1, 100000)


def test_api_returns_jpeg():
    res = client.post(
        "/api/generate",
        files={"file": ("a.png", encode_png(make_image(100, 200)), "image/png")},
        data={"deviceWidth": "390", "deviceHeight": "844"},
    )
    assert res.status_code == 200
    assert res.headers["content-type"] == "image/jpeg"
    assert decode(res.content) is not None


def test_api_returns_400_with_message_for_bad_file():
    res = client.post(
        "/api/generate",
        files={"file": ("a.txt", b"hello", "text/plain")},
        data={"deviceWidth": "390", "deviceHeight": "844"},
    )
    assert res.status_code == 400
    assert "読み込めません" in res.json()["detail"]


def test_index_page_renders():
    res = client.get("/")
    assert res.status_code == 200
    assert 'id="generateForm"' in res.text
