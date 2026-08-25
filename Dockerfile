FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

EXPOSE 8000

# Fix: target src.app.main:app with PYTHONPATH set
ENV PYTHONPATH=/app
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]