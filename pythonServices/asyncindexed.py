import os
import torch
import pandas as pd
import faiss
from transformers import BertTokenizer, AutoModel
from flask import Flask, request, jsonify
import logging
from io import BytesIO
from google.cloud import storage
import numpy as np
import threading

app = Flask(__name__)

# Initialize variables
tokenizer = None
model = None
index = None
answers = None
initialization_completed = False

# Path to the local directory containing the model files
local_model_path = "./indobert-lite-base-p1"


num_layers_to_use = 3  # Number of BERT layers to use for embedding calculation

# Initialization function
def initialize_model_and_data():
    global tokenizer, model, index, answers, initialization_completed

    # Load the tokenizer and model from the local path
    tokenizer = BertTokenizer.from_pretrained(local_model_path)
    model = AutoModel.from_pretrained(local_model_path)

    # Initialize Google Cloud Storage client
    storage_client = storage.Client()

    # Define your bucket and file name
    bucket_name = 'datasetconver'
    blob_name = 'convo-dataset-fix.csv'

    def load_dataset_from_gcs():
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        # Download the blob as bytes
        csv_data = blob.download_as_bytes()
        
        # Use BytesIO to load bytes data into a file-like object
        data = BytesIO(csv_data)
        
        # Load the CSV into a Pandas DataFrame
        dataset = pd.read_csv(data)
        
        return dataset

    # Load dataset
    dataset = load_dataset_from_gcs()
    dataset = dataset.iloc[:, [1, 2]]
    dataset.columns = ['question', 'answer']
    dataset = dataset.dropna(subset=['question', 'answer'])

    questions = dataset["question"].tolist()
    answers = dataset["answer"].tolist()

    # Compute BERT embeddings for all questions
    def compute_embeddings(texts, model, tokenizer, num_layers):
        embeddings = []
        for text in texts:
            input_ids = tokenizer.encode(text, return_tensors='pt')
            with torch.no_grad():
                outputs = model(input_ids, output_hidden_states=True)
                text_embeddings = torch.stack(outputs.hidden_states[-num_layers:]).mean(dim=0).mean(dim=1)
                embeddings.append(text_embeddings.squeeze().numpy())
        return embeddings

    # Compute embeddings for all questions
    precomputed_embeddings = compute_embeddings(questions, model, tokenizer, num_layers_to_use)

    # Build Faiss index for fast similarity search
    d = precomputed_embeddings[0].shape[0]  # Dimensionality of embeddings
    index = faiss.IndexFlatL2(d)
    index.add(np.array(precomputed_embeddings))

    initialization_completed = True

# Start initialization in a background thread
initialization_thread = threading.Thread(target=initialize_model_and_data)
initialization_thread.start()

@app.route("/", methods=["GET"])
def health_check():
    if initialization_completed:
        return jsonify({"message": "Initialization Complete and Code Works!"})
    else:
        return jsonify({"message": "Initialization in Progress..."}), 202

@app.route("/respond", methods=["POST"])
def generate_response():
    if not initialization_completed:
        return jsonify({"error": "Initialization not yet complete"}), 503

    try:
        request_json = request.get_json()
        if request_json and 'user_input' in request_json:
            user_input = request_json['user_input']
            response = find_best_response(user_input)
            return jsonify({"response": response})
        else:
            return jsonify({"error": "Invalid request. Must include 'user_input' in JSON body."}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def find_best_response(input_text):
    global index, answers, tokenizer, model, num_layers_to_use

    # Compute embeddings for user input
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        input_embeddings = torch.stack(outputs.hidden_states[-num_layers_to_use:]).mean(dim=0).mean(dim=1).squeeze().numpy().reshape(1, -1)

    # Search for nearest neighbors in Faiss index
    D, I = index.search(input_embeddings, 1)
    best_response_idx = I[0][0]
    best_response = answers[best_response_idx]

    return best_response

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
