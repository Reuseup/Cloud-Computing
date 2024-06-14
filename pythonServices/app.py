import torch
import pandas as pd
from transformers import BertTokenizer, AutoModel
import torch.nn.functional as F
from google.cloud import storage
from flask import Flask, request, jsonify

app = Flask(__name__)

# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("indobenchmark/indobert-lite-base-p1")
model = AutoModel.from_pretrained("indobenchmark/indobert-lite-base-p1")
model.eval()

num_layers_to_use = 1
model.config.num_hidden_layers = num_layers_to_use

# Initialize Google Cloud Storage client
storage_client = storage.Client()

# Define your bucket and file name
bucket_name = 'datasetconver'
blob_name = 'Conversation-cleaned.csv'

# Function to load dataset from Google Cloud Storage
def load_dataset():
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    dataset = pd.read_csv(blob.download_as_string().decode('utf-8'))
    return dataset

# Initialize dataset as None initially
questions = []
answers = []
dataset_loaded = False

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"message": "Code Works!"})

@app.route("/respond", methods=["POST"])
def generate_response():
    global dataset_loaded, questions, answers

    try:
        if not dataset_loaded:
            dataset = load_dataset()
            questions = dataset["question"].tolist()
            answers = dataset["answer"].tolist()
            dataset_loaded = True

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
    global questions, answers, tokenizer, model, num_layers_to_use

    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        embeddings = torch.stack(outputs.hidden_states[-num_layers_to_use:]).mean(dim=0).mean(dim=1)

    max_similarity = -1
    best_response = ""
    for question, answer in zip(questions, answers):
        question_ids = tokenizer.encode(question, return_tensors='pt')
        with torch.no_grad():
            question_outputs = model(question_ids, output_hidden_states=True)
            question_embeddings = torch.stack(question_outputs.hidden_states[-num_layers_to_use:]).mean(dim=0).mean(dim=1)
            similarity = F.cosine_similarity(embeddings, question_embeddings, dim=1).item()
            if similarity > max_similarity:
                max_similarity = similarity
                best_response = answer

    return best_response

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)

