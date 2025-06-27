from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Load tokenizer and model
model_path = "./bert-tiny-emotion"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

# Emotion label mapping
labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

# Define FastAPI app
app = FastAPI(title="Tiny BERT Emotion Classifier")

# Request model
class TextInput(BaseModel):
    text: str

# POST endpoint
@app.post("/predict")
def predict_emotion(input: TextInput):
    inputs = tokenizer(input.text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
    return {"prediction": labels[pred]}
