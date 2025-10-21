"""
# Documento de Design - Sistema de Pâncreas Artificial

## Decisões Técnicas Fundamentais

### 1. Arquitetura Híbrida PID + ML

**Decisão**: PID como controlador principal, ML como camada de ajuste
**Justificativa**: 
- PID fornece estabilidade e comportamento previsível
- ML adapta parâmetros e melhora previsões ao longo do tempo
- Combinação permite personalização mantendo segurança

**Referências**:
- [Design and Evaluation of a Robust PID Controller for a Fully Implantable Artificial Pancreas](https://pmc.ncbi.nlm.nih.gov/articles/PMC4627627/)
- UVA/Padova Simulator methodology

### 2. Modelo Fisiológico

**Base**: Adaptação do UVA/Padova Simulator
**Componentes**:
- Absorção de glicose gastrointestinal (modelo Dalla Man)
- Farmacocinética da insulina subcutânea
- Dinâmica glicêmica com delays

**Equações principais**:
```
dG/dt = -p1*(G - Gb) - X*G + Ra
dX/dt = -p2*(X - I)
dI/dt = -n*(I - u(t)/Vd)
```
Onde: G=glicose, X=insulina ativa, I=insulina plasmática, u=dose

### 3. Controlador PID Modificado

**Configuração**:
- Kp: 0.6 (resposta proporcional)
- Ki: 0.02 (correção integral lenta)
- Kd: 0.01 (amortecimento derivativo)

**Modificações**:
- Anti-windup com saturação
- Feedback de insulina ativa (IOB)
- Zonas mortas para evitar micro-correções

### 4. Machine Learning - LSTM

**Arquitetura**:
```
Input (6h history) -> LSTM(64) -> Dense(32) -> Output(prediction)
Features: [glucose, insulin, carbs, time_of_day, exercise]
```

**Treinamento**:
- Online learning com mini-batches
- Learning rate adaptativo
- Regularização L2 para evitar overfitting

### 5. Segurança e Limitações

**Hard Limits**:
- Max 5U/hora basal
- Max 4U bolus em 30min
- Detecção spikes >40mg/dL em 15min

**Modo Seguro**:
- Ativado em falhas de sensor
- Somente basal reduzida (50% do perfil)
- Alarmes visuais/sonoros

### 6. Período de Adaptação

**Fase 1 (Dias 1-3)**: Coleta passiva, intervenções mínimas
**Fase 2 (Dias 4-7)**: Ajustes conservadores, confiança crescente  
**Fase 3 (Dias 8-14)**: Autonomia completa baseada em métricas

**Métricas de Confiança**:
- MARD prediction <15%
- Time in range >65%
- Hipoglicemias <2/dia

## Algoritmo Principal

```python
def control_loop(current_glucose, history, ml_model):
    # 1. Calcular IOB (Insulin on Board)
    iob = calculate_iob(history.insulin_doses)
    
    # 2. PID base
    pid_output = pid_controller.calculate(
        target=120, current=current_glucose, iob=iob
    )
    
    # 3. ML prediction e ajuste
    prediction = ml_model.predict(history.features)
    ml_adjustment = calculate_ml_adjustment(prediction, current_glucose)
    
    # 4. Dose final com limites
    final_dose = apply_safety_limits(pid_output + ml_adjustment)
    
    return final_dose, prediction
```
"""