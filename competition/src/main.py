# src/main.py

import os
import yaml
from data_preprocessing.data_loader import load_data
from data_preprocessing.data_cleaner import clean_text
from models.model import create_model
from models.trainer import train_model
from tasks.concept_normalization import normalize_concept
from tasks.elaboration import elaborate_concept
from tasks.extraction_summarization import extract_and_summarize
from tasks.relational_inference import infer_relationship
from tasks.sentiment_analysis import analyze_sentiment
from utils.helper_functions import save_to_json, load_from_json

def main(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    data = load_data(config['data']['file_path'])
    cleaned_data = [clean_text(item) for item in data]

    model, tokenizer = create_model(config['model']['name'])

    if config['train']:
        train_model(model, tokenizer, cleaned_data, cleaned_data, config['output_dir'])

    for task in config['tasks']:
        if task['name'] == 'concept_normalization':
            result = normalize_concept(task['input'])
        elif task['name'] == 'elaboration':
            result = elaborate_concept(task['input'])
        elif task['name'] == 'extraction_summarization':
            result = extract_and_summarize(task['input'])
        elif task['name'] == 'relational_inference':
            result = infer_relationship(task['input'][0], task['input'][1])
        elif task['name'] == 'sentiment_analysis':
            result = analyze_sentiment(task['input'])
        save_to_json(result, os.path.join(config['output_dir'], f"{task['name']}_result.json"))

if __name__ == "__main__":
    main('config.yaml')
