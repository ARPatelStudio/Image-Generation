from fastapi import FastAPI
from pydantic import BaseModel
import torch
from diffusers import AutoPipelineForText2Image
import base64
from io import BytesIO
import os

app = FastAPI()

print("🚀 Loading SDXL-Turbo Model... (isme server start hone par 1-2 minute lagenge)")

# SDXL-Turbo duniya ka fastest model hai (1-4 steps mein image banata hai)
# torch.float32 use kar rahe hain taaki standard Render CPU par chal sake
pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo", 
    torch_dtype=torch.float32
)

# Agar Render par GPU wala tier liya hai toh 'cuda', warna 'cpu'
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe.to(device)
print(f"✅ Model Loaded Successfully on {device.upper()}!")

class ImageRequest(BaseModel):
    prompt: str

@app.post("/generate-image")
def generate_image(req: ImageRequest):
    try:
        # SDXL-Turbo sir 2 steps mein clear image de deta hai (Super fast)
        image = pipe(prompt=req.prompt, num_inference_steps=2, guidance_scale=0.0).images[0]
        
        # Image ko Vertical (Shorts size) mein resize karna
        image = image.resize((720, 1280))
        
        # n8n ke liye Base64 mein convert karna (Taki disk error na aaye)
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return {"status": "success", "image_base64": img_b64}
    
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/")
def home():
    return {"status": "✅ Custom SDXL-Turbo Image Server is Live!"}
