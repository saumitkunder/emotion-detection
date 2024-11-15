# Use Python 3.9 image
FROM python:3.9

# Install necessary system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
    
# Set the working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app /app/app
COPY model /app/model

# Expose Streamlit port
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
