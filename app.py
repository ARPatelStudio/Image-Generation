from fastapi import FastAPI
from pydantic import BaseModel
import torch
from diffusers import AutoPipelineForText2Image
import base64
from io import BytesIO
import os

app = FastAPI()

# ⚠️ NAYA BADLAV: Environment variable se token uthao
HF_TOKEN = os.getenv("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")

print("🚀 Loading SDXL-Turbo Model... (isme server start hone par 1-2 minute lagenge)")

# Token pass kar diya taaki fast download ho aur rate limit ka error na aaye
pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo", 
    torch_dtype=torch.float32,
    token=HF_TOKEN
)

device = "cuda" if torch.cuda.is_available() else "cpu"
pipe.to(device)
print(f"✅ Model Loaded Successfully on {device.upper()}!")

class ImageRequest(BaseModel):
    prompt: str

@app.post("/generate-image")
def generate_image(req: ImageRequest):
    try:
        image = pipe(prompt=req.prompt, num_inference_steps=2, guidance_scale=0.0).images[0]
        image = image.resize((720, 1280))
        
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return {"status": "success", "image_base64": img_b64}
    
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/")
def home():
    return {"status": "✅ Custom SDXL-Turbo Image Server is Live!"}
