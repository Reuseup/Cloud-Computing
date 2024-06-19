# Start from a Python image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the service account key into the container
COPY reuseup-84a02f6e9ef0.json /app/reuseup-84a02f6e9ef0.json

# Set the environment variable to tell Google Cloud SDK where to find the credentials
ENV GOOGLE_APPLICATION_CREDENTIALS="/app/reuseup-84a02f6e9ef0.json"

# Copy the model files into the container
COPY indobert-lite-base-p1 /app/indobert-lite-base-p1

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose port 8080 to match the expected Cloud Run port.
EXPOSE 8080

# Set the environment variable to tell the application what port to listen on.
ENV PORT 8080

# Run the Flask application
CMD ["python", "asyncindexed.py"]
