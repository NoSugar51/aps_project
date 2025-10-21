"""
FROM python:3.11-slim

WORKDIR /app

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements e instalar dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código fonte
COPY src/ src/
COPY config.yaml .
COPY notebooks/ notebooks/

# Criar diretórios necessários
RUN mkdir -p logs data models

# Expor porta para API web
EXPOSE 8000

# Comando padrão
CMD ["python", "src/main.py", "--mode", "simulation", "--web"]
"""