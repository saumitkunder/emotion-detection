# Use a lightweight Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Update pip to avoid dependency resolution issues
RUN pip install --upgrade pip

# Install dependencies
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app/ .

# Expose the Streamlit port
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
