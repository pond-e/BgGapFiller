from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse
import base64
import cv2
import numpy as np
import statistics

def most_used_color(img):
    # https://zenn.dev/kazaki/articles/4bc99a27e33d24
    color_arr = np.vstack(img)
    color_code = ['{:02x}{:02x}{:02x}'.format(*color) for color in color_arr]

    mode = statistics.mode(color_code)
    b = int(mode[0:2], 16)
    g = int(mode[2:4], 16)
    r = int(mode[4:6], 16)
    color_mode = (b, g, r)
    return color_mode


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


@app.post("/uploadfiles/")
async def create_upload_files(request: Request, files: list[UploadFile], deviceWidth: int = Form(), deviceHeight: int = Form()):
    contents = await files[0].read()
    image_data = base64.b64encode(contents).decode("utf-8")

    # from pc-sakan
    nparr = np.fromstring(contents, np.uint8)
    background_img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    background_img_height = background_img.shape[0]
    background_img_width = background_img.shape[1]

    device_width = deviceWidth
    device_height = deviceHeight

    base_height = True
    if background_img_height*device_width > background_img_width*device_height:
        output_width = int(background_img_height * (device_width/device_height))
        output_height = background_img_height
    else:
        output_width = background_img_width
        output_height = int(background_img_width * (device_height/device_width))
        base_height = False

    (output_color_b, output_color_g, output_color_r) = most_used_color(background_img)

    padding_b = np.full((output_height, output_width), output_color_b)
    padding_g = np.full((output_height, output_width), output_color_g)
    padding_r = np.full((output_height, output_width), output_color_r)
    padding = np.stack([padding_b, padding_g, padding_r], 2)

    width_adjust = (output_width//2 + background_img_width//2) - (output_width//2 - background_img_width//2) - background_img_width
    # image synthesis
    if base_height:
        padding[0:background_img_height, (output_width//2 - background_img_width//2):(output_width//2 + background_img_width//2 - width_adjust)] = background_img
    else:
        height_adjust = (output_height//2 + background_img_height//2) - (output_height//2 - background_img_height//2) - background_img_height
        padding[(output_height//2-background_img_height//2):(output_height//2 + background_img_height//2 - height_adjust), 0:background_img_width] = background_img

    output_img_nparr = padding
    _, output_img_tmp = cv2.imencode('.jpeg', output_img_nparr)
    output_img = output_img_tmp.tobytes()
    output_img = base64.b64encode(output_img).decode("utf-8")

    return templates.TemplateResponse("uploadfiles.html", {"request": request, "output_img": output_img})


@app.get("/")
async def main(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})