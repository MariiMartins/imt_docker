# Imagem base com TensorFlow + Python 3.10
FROM tensorflow/tensorflow:2.15.0

# Diretório de trabalho
WORKDIR /app

# Copia e instala dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o restante da aplicação
COPY . .

# Expõe a porta da API
EXPOSE 8002

# Comando padrão
CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8002"]
