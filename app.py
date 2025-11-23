import os
import gdown
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import numpy as np
from PIL import Image
import tensorflow as tf

MODEL_PATH = "image_model.keras"

# Google Drive direct download link (replace this with your file ID)
DRIVE_URL = "https://drive.google.com/file/d/1GqiwgATkYxqPkl9TVmLP81WBe9DDxUzu/view?usp=sharing"

# Download the model automatically if missing
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(DRIVE_URL, MODEL_PATH, quiet=False)

# Load the model
model = tf.keras.models.load_model(MODEL_PATH)
