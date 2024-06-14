# Start from a Python image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the service account key file into the container
COPY reseupbeta-c88ef5a89753.json /app/

# Copy the rest of the application code into the container
COPY . .

# Set the environment variable for Google Cloud credentials
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/reseupbeta-c88ef5a89753.json

# Expose port 5000
EXPOSE 5000

# Run the Flask application
CMD ["python", "app.py"]
