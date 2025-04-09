# Força arquitetura compatível com Apple Silicon (Mac M1/M2)
FROM --platform=linux/amd64 tensorflow/tensorflow:2.15.0

# Define o diretório de trabalho
WORKDIR /app

# Copia e instala as dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o restante da aplicação
COPY . .

# Expõe a porta que o app escuta internamente
EXPOSE 8000

# Comando para iniciar o backend
CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000"]
