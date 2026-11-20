import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_OUTPUT_PIXELS = 50_000_000


class InvalidImageError(ValueError):
    pass


def most_used_color(img: np.ndarray) -> tuple[int, int, int]:
    """BGR 画像で最も多く使われている色を (b, g, r) で返す。"""
    pixels = img.reshape(-1, 3).astype(np.uint32)
    packed = (pixels[:, 0] << 16) | (pixels[:, 1] << 8) | pixels[:, 2]
    values, counts = np.unique(packed, return_counts=True)
    mode = int(values[counts.argmax()])
    return (mode >> 16) & 0xFF, (mode >> 8) & 0xFF, mode & 0xFF


def decode_image(contents: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise InvalidImageError("画像として読み込めませんでした。")
    if img.dtype != np.uint8:
        img = cv2.convertScaleAbs(img, alpha=255.0 / np.iinfo(img.dtype).max)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def fill_gap(contents: bytes, device_width: int, device_height: int) -> bytes:
    """画像を端末の縦横比に合わせ、足りない部分を最頻色で埋めた JPEG を返す。"""
    if device_width <= 0 or device_height <= 0:
        raise InvalidImageError("画面サイズは 1 以上の数値で指定してください。")

    img = decode_image(contents)
    img_height, img_width = img.shape[:2]

    # 画像の方が端末より縦長なら横に、横長なら縦に余白を足す
    if img_height * device_width > img_width * device_height:
        output_width = int(img_height * device_width / device_height)
        output_height = img_height
    else:
        output_width = img_width
        output_height = int(img_width * device_height / device_width)

    if output_width * output_height > MAX_OUTPUT_PIXELS:
        raise InvalidImageError("出力画像が大きすぎます。画面サイズの縦横比を確認してください。")

    canvas = np.empty((output_height, output_width, 3), dtype=np.uint8)
    canvas[:] = most_used_color(img)
    top = (output_height - img_height) // 2
    left = (output_width - img_width) // 2
    canvas[top:top + img_height, left:left + img_width] = img

    ok, encoded = cv2.imencode(".jpg", canvas)
    if not ok:
        raise InvalidImageError("画像の書き出しに失敗しました。")
    return encoded.tobytes()


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.post("/api/generate", response_class=Response)
async def generate(
    file: UploadFile = File(...),
    deviceWidth: int = Form(...),
    deviceHeight: int = Form(...),
):
    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="ファイルサイズは 20MB までです。")

    # OpenCV の処理は CPU を使い続けるので、イベントループを止めないようスレッドプールで実行する
    try:
        jpeg = await run_in_threadpool(fill_gap, contents, deviceWidth, deviceHeight)
    except InvalidImageError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return Response(content=jpeg, media_type="image/jpeg")


@app.get("/")
async def main(request: Request):
    return templates.TemplateResponse(request, "index.html")
