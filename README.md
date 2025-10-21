"""

# Sistema de Pâncreas Artificial (APS) - Simulação e Controle Automático

## ⚠️ AVISOS IMPORTANTES DE SEGURANÇA

**ESTE SISTEMA É DESTINADO EXCLUSIVAMENTE PARA PESQUISA E SIMULAÇÃO.**

- **NÃO UTILIZAR EM PACIENTES REAIS** sem validação clínica completa e aprovação regulatória
- **RISCOS GRAVES**: Hipoglicemia severa, hiperglicemia, cetoacidose diabética
- **RESPONSABILIDADE**: O uso indevido pode resultar em lesões graves ou morte
- Sistema em desenvolvimento - não validado clinicamente

## Visão Geral

Sistema de Pâncreas Artificial híbrido combinando:

- **Controlador PID** como base (inspirado no UVA/Padova Simulator)
- **Machine Learning** (LSTM) para ajustes e previsões
- **Período de adaptação** automático (7-14 dias)
- **Simulação robusta** com aceleração temporal
- **Preparação para hardware real** (Dana + FreeStyle Libre 2)

## Características Principais

### Controle

- PID com anti-windup e saturação
- Camada basal adaptativa
- Bolus automático com salvaguardas
- ML para ajuste de ganhos e previsão

### Segurança

- Hard-limits configuráveis
- Detecção de falhas de sensor
- Logs auditáveis
- Modo seguro automático

### Simulação

- Aceleração temporal (1x-100x)
- Timeline personalizável (6h-90d)
- Catálogo de alimentos
- Cenários de teste reproducíveis

## Instalação e Execução

### Pré-requisitos

```bash
# 1. Instalar Git para Windows
# Baixe o instalador em: https://git-scm.com/download/win
# Execute o instalador e reinicie o PowerShell/Terminal

# 2. Verificar instalação do Git
git --version

# 3. Configurar Git (primeira vez)
git config --global user.name "Seu Nome"
git config --global user.email "seu.email@exemplo.com"

# 4. Inicializar repositório
cd c:\Users\vicia\Desktop\TCC\Software\aps_project
git init
git add .
git commit -m "Inicialização do projeto"

# 5. Instalar dependências Python
pip install -r requirements.txt
```

### Execução em Modo Simulação

```bash
# Com interface web
python -m src.main --mode simulation --web
# Acesse http://127.0.0.1:8000 ou http://localhost:8000 no navegador
```

### Configuração

Edite `config.yaml` com parâmetros do paciente:

```yaml
patient:
  age: 30
  weight_kg: 75
  carb_ratio: 10  # g carb por unidade insulina
  isf: 50         # mg/dL reduzidos por unidade
```

## Arquitetura

```
src/
├── main.py                 # Ponto de entrada
├── controller/            # Algoritmos de controle
│   ├── pid_controller.py  # PID principal
│   └── hybrid_controller.py # PID + ML
├── ml/                    # Machine Learning
│   ├── model.py          # LSTM para previsão
│   └── trainer.py        # Treinamento incremental
├── sim/                   # Simulação
│   ├── simulator.py      # Motor de simulação
│   └── food_catalog.py   # Base de alimentos
└── hardware/             # Integração hardware
    ├── dana_adapter.py   # Bomba Dana
    └── libre2_adapter.py # Sensor Libre 2
```

## Métricas de Avaliação

- **Tempo em faixa** (70-180 mg/dL): >70%
- **Hipoglicemia** (<70 mg/dL): <4%
- **RMSE predição**: <15 mg/dL
- **Overshoot médio**: <40 mg/dL

## Desenvolvimento e Testes

```bash
# Testes automatizados
pytest tests/

# Notebook de análise
jupyter notebook notebooks/analysis_example.ipynb

# Docker
docker build -t aps-system .
docker run -p 8000:8000 aps-system
```

## Referências Científicas

- UVA/Padova T1D Simulator (FDA approved)
- PID Control in Artificial Pancreas Systems
- Hybrid ML-PID Approaches for Glucose Control

## Suporte

Para questões técnicas, consulte DESIGN.md
Para configuração de hardware real, veja hardware/README.md
"""
