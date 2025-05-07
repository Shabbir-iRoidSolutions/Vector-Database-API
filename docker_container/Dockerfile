# Use official Python image as a base
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install git and system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy local code into the container
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Expose the port your app runs on
EXPOSE 2100

# Run the application
CMD ["python", "app.py"]
