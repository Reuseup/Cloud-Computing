import os
import torch
import pandas as pd
import faiss
import numpy as np
import logging
import threading

from transformers import BertTokenizer, AutoModel
from flask import Flask, request, jsonify
from io import BytesIO
from google.cloud import storage
from google.api_core.exceptions import GoogleAPIError
from pandas.errors import EmptyDataError, ParserError

app = Flask(__name__)

# Initialize variables
tokenizer = None
model = None
index = None
answers = None
initialization_completed = False
initialization_failed = False
initialization_error_message = None
num_layers_to_use = 3  # Number of BERT layers to use for embedding calculation

# Initialization function
def initialize_model_and_data():
    global tokenizer, model, index, answers, initialization_completed, initialization_failed, initialization_error_message

    try:
        # Load the tokenizer and model from the local directory
        tokenizer = BertTokenizer.from_pretrained("/app/indobert-lite-base-p1")
        model = AutoModel.from_pretrained("/app/indobert-lite-base-p1")

        # Initialize Google Cloud Storage client
        storage_client = storage.Client()

        # Define your bucket and file name
        bucket_name = 'datasetconver'
        blob_name = 'convo-dataset-fix.csv'

        def load_dataset_from_gcs():
            try:
                bucket = storage_client.bucket(bucket_name)
                blob = bucket.blob(blob_name)
                
                # Download the blob as bytes
                csv_data = blob.download_as_bytes()
                
                # Use BytesIO to load bytes data into a file-like object
                data = BytesIO(csv_data)
                
                # Load the CSV into a Pandas DataFrame
                dataset = pd.read_csv(data)
                
                # Optional: Additional data checks or processing can be done here
                if dataset.empty:
                    raise ValueError("The dataset is empty.")
                
                return dataset

            except GoogleAPIError as e:
                logging.error(f"Error accessing Google Cloud Storage: {e}")
                raise RuntimeError("Failed to access the dataset from Google Cloud Storage.")
            
            except (EmptyDataError, ParserError) as e:
                logging.error(f"Error parsing the CSV file: {e}")
                raise ValueError("Failed to parse the CSV file.")
            
            except Exception as e:
                logging.error(f"Unexpected error: {e}")
                raise RuntimeError("An unexpected error occurred while loading the dataset.")

        def compute_embeddings(texts, model, tokenizer, num_layers):
            embeddings = []
            for text in texts:
                try:
                    input_ids = tokenizer.encode(text, return_tensors='pt')
                    with torch.no_grad():
                        outputs = model(input_ids, output_hidden_states=True)
                        text_embeddings = torch.stack(outputs.hidden_states[-num_layers:]).mean(dim=0).mean(dim=1)
                        embeddings.append(text_embeddings.squeeze().numpy())
                except Exception as e:
                    logging.error(f"Error computing embeddings for text '{text}': {e}")
                    raise RuntimeError("Failed to compute embeddings for the provided text.")
            return embeddings

        # Load dataset
        dataset = load_dataset_from_gcs()
        dataset = dataset.iloc[:, [1, 2]]
        dataset.columns = ['question', 'answer']
        dataset = dataset.dropna(subset=['question', 'answer'])

        questions = dataset["question"].tolist()
        answers = dataset["answer"].tolist()

        # Compute embeddings for all questions
        precomputed_embeddings = compute_embeddings(questions, model, tokenizer, num_layers_to_use)

        # Build Faiss index for fast similarity search
        d = precomputed_embeddings[0].shape[0]  # Dimensionality of embeddings
        index = faiss.IndexFlatL2(d)
        index.add(np.array(precomputed_embeddings))

        initialization_completed = True

    except Exception as e:
        initialization_failed = True
        initialization_error_message = str(e)
        logging.error(f"Initialization failed: {e}")

# Start initialization in a background thread
initialization_thread = threading.Thread(target=initialize_model_and_data)
initialization_thread.start()

@app.route("/", methods=["GET"])
def health_check():
    if initialization_failed:
        return jsonify({"message": "Initialization Failed", "error": initialization_error_message}), 500
    elif initialization_completed:
        return jsonify({"message": "Initialization Complete and Code Works!"})
    else:
        return jsonify({"message": "Initialization in Progress..."}), 202

@app.route("/respond", methods=["POST"])
def generate_response():
    if not initialization_completed:
        if initialization_failed:
            return jsonify({"error": "Initialization failed", "message": initialization_error_message}), 500
        else:
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
        logging.error(f"Error handling request: {e}")
        return jsonify({"error": str(e)}), 500

def find_best_response(input_text):
    global index, answers, tokenizer, model, num_layers_to_use

    try:
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

    except Exception as e:
        logging.error(f"Error finding the best response: {e}")
        raise RuntimeError("Failed to find the best response for the provided input.")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
