from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

import numpy as np
from PIL import Image
import tensorflow as tf

# Load your model (upload it to server root later)
model = tf.keras.models.load_model("image_model.keras")

app = FastAPI()

# Templates & static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def preprocess(img):
    img = img.resize((224, 224))   # Set your input size
    img = np.array(img) / 255.0
    img = np.expand_dims(img, 0)
    return img

def convert_to_label(pred):
    """Convert model output to High / Medium / Low."""
    score = float(pred[0][0])  # Assumes 1 output neuron

    if score >= 0.7:
        return "High Score"
    elif score >= 0.4:
        return "Medium Score"
    else:
        return "Low Score"

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "prediction": None})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    img = Image.open(file.file).convert("RGB")
    x = preprocess(img)
    pred = model.predict(x)

    label = convert_to_label(pred)

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "prediction": label}
    )
