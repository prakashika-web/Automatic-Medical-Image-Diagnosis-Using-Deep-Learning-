from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import io

app = FastAPI()

# Allow frontend to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load model
model = models.resnet34(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)
checkpoint = torch.load("model_weights.pth", map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        conf = probs[0][pred].item()

    return {
        "prediction":"Diseased" if pred == 1 else "Healthy",
        "confidence": round(conf * 100, 2)
    }
if __name__== "__main__":
    import uvicorn
    uvicorn.run("app:app",
                host="127.0.0.1",port=8000,
                reload=True)
