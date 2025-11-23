from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

import numpy as np
from PIL import Image
import tensorflow as tf

# TEMPORARY FIX: Disable model loading
model = None

app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def preprocess(img):
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, 0)
    return img

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "prediction": None}
    )

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    
    # TEMPORARY WARNING (until model is uploaded)
    if model is None:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "prediction": "MODEL NOT LOADED — Upload image_model.keras to server."}
        )

    img = Image.open(file.file).convert("RGB")
    x = preprocess(img)
    pred = model.predict(x)

    # Convert pred to High/Medium/Low
    score = float(pred[0][0])
    if score >= 0.7:
        label = "High Score"
    elif score >= 0.4:
        label = "Medium Score"
    else:
        label = "Low Score"

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "prediction": label}
    )
