# Use Python latest slim image as base
FROM python:slim

# Build command for Linux AMD64:
# docker build --platform linux/amd64 -t khoa0702/langchain-neo4j-api .

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port 8000 (FastAPI default port)
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"] 