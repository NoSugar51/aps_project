#!/bin/bash
echo "🚀 Instalando dependências do Sistema APS..."

# Criar ambiente virtual se não existir
if [ ! -d "venv" ]; then
    echo "📦 Criando ambiente virtual..."
    python -m venv venv
fi

# Ativar ambiente virtual
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Atualizar pip
pip install --upgrade pip

# Instalar dependências
echo "📚 Instalando dependências Python..."
pip install -r requirements.txt

# Criar diretórios necessários
echo "📁 Criando estrutura de diretórios..."
mkdir -p data logs models exports

echo "✅ Instalação concluída!"
echo "Para ativar: source venv/bin/activate"
echo "Para executar: python src/main.py --help"