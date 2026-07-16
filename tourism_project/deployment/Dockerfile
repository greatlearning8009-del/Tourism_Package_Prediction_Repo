# Optional: build/run the app in a container outside of Codespaces.
#   docker build -t tourism-app -f tourism_project/deployment/Dockerfile .
#   docker run --rm -p 8501:8501 tourism-app
FROM python:3.11-slim

WORKDIR /app

# Install deps first for better layer caching
COPY tourism_project/deployment/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the project (need the models dir for the app to load the model)
COPY tourism_project /app/tourism_project

EXPOSE 8501

CMD ["streamlit", "run", "tourism_project/deployment/app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.enableXsrfProtection=false"]
