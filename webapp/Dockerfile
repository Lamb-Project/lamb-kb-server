FROM python:3.11-slim-bookworm
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt ./requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY . .

# Create static directory
RUN mkdir -p static

# Create templates directory
RUN mkdir -p templates

# Create default .env file if one doesn't exist
RUN touch .env

# Expose the port the app runs on
EXPOSE 9091

# Command to run the web application
CMD ["python", "run_webapp.py"]