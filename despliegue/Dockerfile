FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9-slim
WORKDIR /app
EXPOSE 80
COPY requirements.txt .
RUN ["pip", "install", "-r","./requirements.txt"]
COPY . .
CMD ["python3","main.py"]