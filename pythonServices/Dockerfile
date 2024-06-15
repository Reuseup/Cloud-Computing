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

# Expose port 8080 to match the expected Cloud Run port.
EXPOSE 8080

# Set the environment variable to tell the application what port to listen on.
ENV PORT 8080

# Run the Flask application
CMD ["python", "indexed.py"]
