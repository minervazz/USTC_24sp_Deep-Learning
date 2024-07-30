# src/models/model.py

from transformers import AutoModelForSequenceClassification, AutoTokenizer

def create_model(model_name):
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer
