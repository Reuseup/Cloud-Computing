
# **IndoBERT-based Conversation API**

## Table of Contents

- [About the Project](#about-the-project)
- [Key Features](#key-features)
- [Technology Stack](#technology-stack)
- [Setup and Installation](#setup-and-installation)
  - [Prerequisites](#prerequisites)
  - [Clone the Repository](#clone-the-repository)
  - [Docker Setup](#docker-setup)
  - [Manual Setup](#manual-setup)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)

## About the Project

This project is a **Flask-based** web service for a question-and-answer system powered by **IndoBERT**, a transformer model tailored for the Indonesian language. It leverages pre-trained BERT embeddings and **Faiss** for efficient similarity search, providing responses to user queries based on pre-defined answers stored in a dataset.

## Key Features

- **Pre-trained IndoBERT Model**: Uses the IndoBERT model for embedding textual data.
- **Faiss Indexing**: Efficient similarity search using Faiss for quick response retrieval.
- **Google Cloud Storage Integration**: Loads dataset securely from Google Cloud Storage.
- **Flask API**: Exposes endpoints for health checks and querying the model.

## Technology Stack

- **Python**: Core language for the service.
- **Flask**: Web framework used to serve the API.
- **Transformers**: For loading and using the BERT model.
- **Faiss**: Efficient similarity search.
- **Google Cloud Storage**: Secure storage and retrieval of the dataset.
- **Docker**: Containerization of the application for consistent deployment.

## Setup and Installation

### Prerequisites

Before you begin, ensure you have the following installed:

- **Docker**: For running the application in a containerized environment.
- **Python 3.10**: If you choose to run the application without Docker.
- **Google Cloud SDK**: For managing Google Cloud services if not using Docker.

### Clone the Repository

Clone the repository to your local machine:

```bash
git clone https://github.com/your_username/repository.git
cd repository
```

### Docker Setup

The recommended way to run the application is using Docker. Follow these steps to set up the project:

1. **Build the Docker image**:

   ```bash
   docker build -t indobert-qa-service .
   ```

2. **Run the Docker container**:

   ```bash
   docker run -p 8080:8080 indobert-qa-service
   ```

The application will be accessible at `http://localhost:8080`.

### Manual Setup

If you prefer to run the application without Docker, follow these steps:

1. **Set up a virtual environment**:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install the dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Set the environment variables** (Replace with actual paths/values):

   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-key.json"
   export PORT=8080
   ```

4. **Run the application**:

   ```bash
   python asyncindexed.py
   ```

## Usage

Once the application is running, you can interact with it using the provided API endpoints.

### API Endpoints

- **Health Check**: `GET /`
  - Check the status of the service.
  - **Response**:
    ```json
    {
      "message": "Initialization Complete and Code Works!"
    }
    ```

- **Generate Response**: `POST /respond`
  - Retrieve the best matching response to a user query.
  - **Request**:
    ```json
    {
      "user_input": "Your question here"
    }
    ```

## Contributing

We welcome contributions to improve this project! To contribute:

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

For major changes, please open an issue first to discuss what you would like to change.

## Acknowledgments

- **Hugging Face** for providing the BERT model and the Transformers library.
- **Google Cloud** for their storage services.
- **Faiss** for the efficient similarity search library.
