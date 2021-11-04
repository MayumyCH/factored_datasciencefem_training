# FROM <> Imagen padre en la cual nos basaremos (contiene Dependencias)
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9-slim

# WORKDIR <> Para especificar nuestro directorio de trabajo
WORKDIR /app

# El puerto que vamos a exponer
EXPOSE 80

# COPY <> Copiar de nuestro Host (Maquina) a la imagen de Docker
# Para traer el archivo requirements.txt al Dockerfile.
COPY requirements.txt .

# RUN <> Compilar el codigo incluyendo algunas dependencias
RUN ["pip", "install", "-r","./requirements.txt"]

COPY . .

# CMD . . <> Comandos a correr para iniciar el servicio 
CMD ["python3","main.py"]