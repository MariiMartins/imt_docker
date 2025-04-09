# Imagem base com Python 3.12
FROM python:3.12-slim

# Diretório de trabalho
WORKDIR /app

# Evita perguntas interativas no apt
ENV DEBIAN_FRONTEND=noninteractive

# Instala dependências do sistema (essenciais para PIL e streamlit)
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copia e instala as dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o restante da aplicação
COPY . .

# Expõe a porta do frontend
EXPOSE 8501

# Comando padrão
CMD ["streamlit", "run", "frontend.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
