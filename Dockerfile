# 1. Base image
FROM python:3.10-slim

# 2. Set working directory
WORKDIR /app

# 3. Copy files
COPY requirements.txt .
COPY app.py .
COPY ui.py .
COPY M3.pdf .

# 4. Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Expose Streamlit port
EXPOSE 8501

# 6. Run Streamlit app
CMD ["streamlit", "run", "ui.py", "--server.port=8501", "--server.address=0.0.0.0"]
