"""
Controlador Híbrido PID + Machine Learning para Sistema de Pâncreas Artificial

Combina:
- Controlador PID como base estável e confiável
- Rede neural LSTM para ajustes e previsões
- Sistema de confiança adaptativo
- Salvaguardas de segurança integradas

Referências:
- Hybrid closed-loop artificial pancreas systems
- Machine learning enhanced PID control
- Safety-critical ML in medical devices
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import List

logger = logging.getLogger(__name__)

class ControlMode(Enum):
    """Modos de operação do controlador"""
    LEARNING = "learning"      # Período de adaptação inicial
    CONSERVATIVE = "conservative"  # Operação conservadora
    AUTONOMOUS = "autonomous"     # Operação totalmente autônoma
    SAFETY_ONLY = "safety_only"  # Modo seguro (emergência)

@dataclass
class ControlOutput:
    """Saída completa do controlador híbrido"""
    dose: float  # Dose final em unidades
    dose_type: str
    prediction: List[float] = field(default_factory=list)  # Previsões futuras
    confidence: float = 0.0  # Confiança do sistema (0-1)
    pid_component: float = 0.0  # Contribuição do PID
    ml_component: float = 0.0   # Ajuste do ML

class HybridController:
    """
    Controlador híbrido que combina PID clássico com Machine Learning
    
    Arquitetura:
    1. PID fornece controle base estável
    2. ML prediz tendências e ajusta parâmetros
    3. Sistema de confiança gradual durante adaptação
    4. Múltiplas camadas de segurança
    """
    
    def __init__(self, config: dict, ml_model, safety_monitor):
        """
        Inicializa controlador híbrido
        
        Args:
            config: Configuração do controlador
            ml_model: Modelo ML para previsões
            safety_monitor: Monitor de segurança
        """
        self.config = config
        self.ml_model = ml_model
        self.safety_monitor = safety_monitor
        
        # Modo inicial
        self.current_mode = ControlMode.LEARNING
    
    def calculate_dose(self, current_glucose, glucose_history, insulin_history, meal_history) -> ControlOutput:
        """
        Calcula dose de insulina usando abordagem híbrida
        
        Args:
            current_glucose: Glicemia atual em mg/dL
            glucose_history: Histórico de glicemia
            insulin_history: Histórico de insulina
            meal_history: Histórico de refeições
            
        Returns:
            ControlOutput com decisão final
        """
        try:
            # 1. Avaliar modo atual do sistema
            self._update_control_mode()
            
            # 2. Calcular IOB
            current_iob = 0  # Placeholder para cálculo de IOB
            
            # 3. Verificações de segurança prioritárias
            safety_check = self.safety_monitor.check_safety(
                glucose=current_glucose,
                insulin_history=insulin_history,
                sensor_data={'current': current_glucose, 'history': glucose_history}
            )
            
            if not safety_check.is_safe:
                return self._create_safety_override_output(safety_check)
            
            # 4. Calcular componente PID (placeholder)
            pid_output = 1.0  # Chamada para cálculo PID deve ser aqui
            
            # 5. Calcular componente ML (se confiança suficiente)
            ml_adjustment = 0.0
            prediction = []
            ml_confidence = 0.0
            
            if self._should_use_ml():
                ml_result = self._calculate_ml_component(
                    current_glucose, glucose_history, insulin_history, meal_history
                )
                ml_adjustment = ml_result['adjustment']
                prediction = ml_result['prediction']
                ml_confidence = ml_result['confidence']
            
            # 6. Combinar PID + ML
            final_dose = pid_output * (1 + ml_adjustment)
            
            # 7. Aplicar limites finais de segurança
            final_dose = min(final_dose, self.config['max_dose'])
            
            # 8. Criar saída final
            output = ControlOutput(
                dose=final_dose,
                dose_type='bolus',
                prediction=prediction,
                confidence=self._calculate_system_confidence(),
                pid_component=pid_output,
                ml_component=ml_adjustment
            )
            
            return output
            
        except Exception as e:
            logger.error(f"Erro no controlador híbrido: {e}")
            return ControlOutput(0, 'none', 0, 0, 0, 0)
    
    def _update_control_mode(self):
        """Atualiza modo de controle baseado no tempo de operação e métricas"""
        pass  # Implementar lógica de atualização de modo
    
    def _should_use_ml(self) -> bool:
        """Determina se deve usar componente ML baseado no modo atual"""
        return self.current_mode == ControlMode.AUTONOMOUS
    
    def _calculate_ml_component(self, glucose, glucose_history, insulin_history, meal_history) -> dict:
        """
        Calcula ajuste baseado em ML
        
        Returns:
            Dict com adjustment, prediction e confidence
        """
        try:
            # Preparar features para o modelo
            features = self._prepare_ml_features(
                glucose, glucose_history, insulin_history, meal_history
            )
            
            # Fazer previsão
            prediction_result = self.ml_model.predict(features)
            
            # Calcular ajuste baseado na previsão
            predicted_glucose = prediction_result['predictions'][-1]  # Última previsão
            adjustment = (predicted_glucose - glucose) / 100.0  # Ajuste proporcional
            
            return {
                'adjustment': adjustment,
                'prediction': prediction_result['predictions'],
                'confidence': prediction_result['confidence']
            }
            
        except Exception as e:
            logger.error(f"Erro no componente ML: {e}")
            return {'adjustment': 0.0, 'prediction': [], 'confidence': 0.0}
    
    def _prepare_ml_features(self, glucose, glucose_history, insulin_history, meal_history) -> np.ndarray:
        """Prepara features para o modelo ML"""
        # Implementação simplificada - expandir conforme necessário
        features = []
        
        # Features de glicemia (últimas 6 horas)
        if glucose_history:
            recent_glucose = [g['value'] for g in glucose_history[-36:]]  # 6h
            while len(recent_glucose) < 36:
                recent_glucose.insert(0, recent_glucose[0] if recent_glucose else glucose)
            features.extend(recent_glucose)
        else:
            features.extend([glucose] * 36)
        
        # Features de insulina
        recent_insulin = 0  # Placeholder para cálculo de IOB
        features.append(recent_insulin)
        
        # Features temporais
        now = np.datetime64('now')
        features.extend([
            now.astype('datetime64[h]').astype(int) % 24 / 24.0,  # Hora do dia normalizada
            now.astype('datetime64[D]').astype(int) % 7 / 6.0,  # Dia da semana normalizado
        ])
        
        # Feature de refeição recente
        recent_carbs = 0
        for meal in meal_history:
            if (now - meal['timestamp']).astype('timedelta64[m]').astype(int) < 120:
                recent_carbs += meal.get('carbs', 0)
        features.append(min(recent_carbs / 100.0, 1.0))  # Normalizar
        
        return np.array(features).reshape(1, -1)
    
    def _calculate_system_confidence(self) -> float:
        """
        Calcula confiança geral do sistema baseado em métricas históricas
        
        Returns:
            Confiança entre 0 e 1
        """
        return 0.85  # Implementar cálculo real de confiança
    
    def _create_safety_override_output(self, safety_check) -> ControlOutput:
        """Cria saída de emergência para situações inseguras"""
        return ControlOutput(
            dose=0.0,
            dose_type='basal',
            prediction=[],
            confidence=0.0,
            pid_component=0.0,
            ml_component=0.0,
            reasoning=f"🚨 OVERRIDE SEGURANÇA: {safety_check.message}",
            safety_override=True
        )
    
    def get_current_basal_rate(self) -> float:
        """
        Obtém taxa basal atual baseada no perfil e horário
        
        Returns:
            Taxa basal em U/h
        """
        return 1.0  # Placeholder, implementar lógica real
    
    def update_ml_model(self, new_model):
        """
        Atualiza modelo ML em tempo real
        
        Args:
            new_model: Novo modelo treinado
        """
        self.ml_model = new_model
        logger.info("Modelo ML atualizado")
    
    def force_mode_change(self, new_mode: ControlMode):
        """
        Força mudança de modo (para testes ou emergências)
        
        Args:
            new_mode: Novo modo de operação
        """
        old_mode = self.current_mode
        self.current_mode = new_mode
        logger.warning(f"Modo forçado: {old_mode.value} → {new_mode.value}")
    
    def get_performance_metrics(self, hours: int = 24) -> dict:
        """
        Calcula métricas de desempenho do controlador híbrido
        
        Args:
            hours: Período para análise
            
        Returns:
            Métricas calculadas
        """
        return {}  # Placeholder, implementar cálculo real

# Funções utilitárias para o controlador híbrido

def calculate_insulin_sensitivity_factor(weight_kg: float, age: int) -> float:
    """
    Calcula ISF (Insulin Sensitivity Factor) baseado em características do paciente
    
    Baseado na regra de 1800 ajustada por idade:
    ISF = 1800 / (peso * fator_idade)
    
    Args:
        weight_kg: Peso em kg
        age: Idade em anos
        
    Returns:
        ISF em mg/dL por unidade de insulina
    """
    # Fator de correção por idade
    if age < 20:
        age_factor = 1.2  # Jovens mais sensíveis
    elif age > 50:
        age_factor = 0.8  # Mais velhos menos sensíveis
    else:
        age_factor = 1.0
    
    isf = (1800 / weight_kg) * age_factor
    
    # Limites de segurança
    isf = np.clip(isf, 20, 100)  # Entre 20-100 mg/dL/U
    
    return isf

def calculate_carb_ratio(weight_kg: float, isf: float) -> float:
    """
    Calcula razão carboidrato:insulina
    
    Baseado na regra de 500:
    CR = 500 / TDD, onde TDD ≈ peso/4
    
    Args:
        weight_kg: Peso em kg
        isf: Fator de sensibilidade à insulina
        
    Returns:
        Gramas de carboidrato por unidade de insulina
    """
    # Estimativa de dose diária total
    tdd = weight_kg / 4.0
    
    # Regra de 500
    carb_ratio = 500 / tdd
    
    # Ajuste por ISF (correlação empírica)
    carb_ratio *= (isf / 50.0) ** 0.3
    
    # Limites de segurança
    carb_ratio = np.clip(carb_ratio, 5, 30)  # Entre 5-30 g/U
    
    return carb_ratio

def estimate_meal_bolus(carbs: float, carb_ratio: float, current_glucose: float,
                       target_glucose: float, isf: float, iob: float) -> float:
    """
    Calcula bolus para refeição usando método padrão
    
    Args:
        carbs: Gramas de carboidrato
        carb_ratio: Razão carboidrato:insulina
        current_glucose: Glicemia atual
        target_glucose: Glicemia alvo
        isf: Fator de sensibilidade
        iob: Insulina ativa
        
    Returns:
        Dose de bolus em unidades
    """
    # Bolus para carboidrato
    carb_bolus = carbs / carb_ratio
    
    # Correção para glicemia atual
    glucose_correction = (current_glucose - target_glucose) / isf
    
    # Bolus total ajustado por IOB
    total_bolus = carb_bolus + glucose_correction - iob
    
    # Nunca negativo
    return max(0, total_bolus)