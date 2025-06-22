from fastapi import FastAPI, HTTPException, Header, Request
from pydantic import BaseModel
from typing import List, Optional
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import requests
from io import BytesIO
import pickle
import os

API_KEY = "abc"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def download_file(url: str, save_path: str):
    if not os.path.exists(save_path):
        print(f"Downloading {url} → {save_path}")
        response = requests.get(url)
        with open(save_path, "wb") as f:
            f.write(response.content)
        print("Download complete.")

download_file(
    "https://drive.google.com/file/d/1RiR5oNhiVa2vlGvxl6pCO5NBs-TPa-bn/view?usp=sharing",
    "category_bert_model/model.safetensors"
)

text_model_path = "category_bert_model"
text_model = BertForSequenceClassification.from_pretrained(text_model_path)
text_tokenizer = BertTokenizer.from_pretrained(text_model_path)
text_model.to(DEVICE)
text_model.eval()

with open("category_label_encoder.pkl", "rb") as f:
    category_label_encoder = pickle.load(f)

class_names = [
    "ak", "pistol", "rifle", "helmet", "pounch",
    "shutgun", "vest", "M series", "mashinegun", "backpack", "other"
]

def load_image_model(path='weapon_classifier.pth'):
    checkpoint = torch.load(path, map_location=DEVICE)
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    return model

image_model = load_image_model()

image_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

app = FastAPI(title="Category & Subcategory Classifier API")


class Photo(BaseModel):
    photo_id: str
    url: str

class PostRequest(BaseModel):
    post_id: str
    text: str
    photos: List[Photo]

class PredictionResponse(BaseModel):
    object_id: str
    category: str
    subcategory: str
    confidence: float
    photo_ids: List[str]

class APIResponse(BaseModel):
    post_id: str
    predictions: List[PredictionResponse]

def predict_category(text: str) -> str:
    inputs = text_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = text_model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        confidence, pred = torch.max(probs, dim=1)
        category = category_label_encoder.inverse_transform([pred.item()])[0]
    return category, confidence.item()


def predict_image_subcategory(url: str) -> Optional[tuple]:
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content)).convert('RGB')
        img_tensor = image_transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outputs = image_model(img_tensor)
            probs = F.softmax(outputs, dim=1)
            conf, preds = torch.max(probs, 1)
        return class_names[preds.item()], conf.item()
    except Exception as e:
        print(f"Image prediction failed: {e}")
        return None, None


# === API Endpoint ===
@app.post("/predict_batch", response_model=List[APIResponse])
async def predict_batch(data: List[PostRequest], authorization: Optional[str] = Header(None)):

    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    results = []

    for post in data:
        category, text_conf = predict_category(post.text)

        predictions = []
        for photo in post.photos:
            subcategory, img_conf = predict_image_subcategory(photo.url)
            if subcategory is not None:
                predictions.append(PredictionResponse(
                    object_id=photo.photo_id,
                    category=category,
                    subcategory=subcategory,
                    confidence=round(img_conf, 4),
                    photo_ids=[photo.photo_id]
                ))

        results.append(APIResponse(post_id=post.post_id, predictions=predictions))

    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
