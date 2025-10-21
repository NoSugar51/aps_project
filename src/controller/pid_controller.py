"""
Controlador PID para Sistema de Pâncreas Artificial

Baseado em literatura científica sobre controle PID em sistemas APS:
- Design and Evaluation of a Robust PID Controller for a Fully Implantable Artificial Pancreas
- UVA/Padova Type 1 Diabetes Simulator methodology

Características:
- PID clássico com anti-windup
- Feedback de insulina ativa (IOB)
- Zonas mortas para estabilidade
- Saturação de saída configurável
"""

import logging
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class PIDState:
    """Estado interno do controlador PID"""
    error: float = 0.0
    previous_error: float = 0.0
    integral: float = 0.0
    derivative: float = 0.0
    last_output: float = 0.0
    last_time: Optional[datetime] = None

@dataclass
class PIDOutput:
    """Saída do controlador PID"""
    dose: float  # Unidades de insulina
    p_component: float  # Componente proporcional
    i_component: float  # Componente integral
    d_component: float  # Componente derivativo
    saturated: bool = False  # Se a saída foi saturada
    reason: str = ""  # Explicação da decisão

class PIDController:
    """
    Controlador PID especializado para entrega de insulina
    
    Implementa PID clássico com modificações para segurança:
    - Anti-windup com saturação
    - Feedback de insulina ativa
    - Zona morta para evitar micro-oscilações
    """
    
    def __init__(self, kp: float = 0.6, ki: float = 0.02, kd: float = 0.01,
                 target_glucose: float = 120, max_output: float = 5.0,
                 anti_windup_limit: float = 2.0, dead_zone: float = 10.0):
        """
        Inicializa controlador PID
        
        Args:
            kp: Ganho proporcional (padrão baseado em literatura)
            ki: Ganho integral (baixo para evitar overshoot)
            kd: Ganho derivativo (para amortecimento)
            target_glucose: Glicemia alvo em mg/dL
            max_output: Saída máxima em U/h
            anti_windup_limit: Limite para anti-windup
            dead_zone: Zona morta em mg/dL (evita micro-correções)
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.target_glucose = target_glucose
        self.max_output = max_output
        self.anti_windup_limit = anti_windup_limit
        self.dead_zone = dead_zone
        
        # Estado interno
        self.state = PIDState()
        self.state.last_time = datetime.now()
        
        # Histórico para análise
        self.history = []
        
        logger.info(f"PID Controller inicializado: Kp={kp}, Ki={ki}, Kd={kd}")
    
    def calculate(self, current_glucose: float, iob: float = 0.0,
                  dt: Optional[float] = None) -> PIDOutput:
        """
        Calcula saída do controlador PID
        
        Args:
            current_glucose: Glicemia atual em mg/dL
            iob: Insulina ativa (Insulin on Board) em unidades
            dt: Intervalo de tempo em minutos (None para calcular automaticamente)
            
        Returns:
            PIDOutput com dose calculada e componentes
        """
        current_time = datetime.now()
        
        # Calcular intervalo de tempo
        if dt is None:
            if self.state.last_time is not None:
                dt = (current_time - self.state.last_time).total_seconds() / 60.0
            else:
                dt = 10.0  # Padrão: 10 minutos
        
        # Calcular erro (considerando IOB como feedback)
        error = current_glucose - self.target_glucose
        
        # Aplicar zona morta
        if abs(error) < self.dead_zone:
            error = 0.0
        
        # Componente Proporcional
        p_component = self.kp * error
        
        # Componente Integral com anti-windup
        if not self._is_saturated():
            self.state.integral += error * dt
            # Limitar integral para evitar windup
            self.state.integral = np.clip(
                self.state.integral, 
                -self.anti_windup_limit, 
                self.anti_windup_limit
            )
        
        i_component = self.ki * self.state.integral
        
        # Componente Derivativo
        if dt > 0:
            derivative = (error - self.state.previous_error) / dt
            # Filtro simples para ruído
            self.state.derivative = 0.8 * self.state.derivative + 0.2 * derivative
        
        d_component = self.kd * self.state.derivative
        
        # Saída PID bruta
        pid_output = p_component + i_component + d_component
        
        # Aplicar saturação de segurança
        saturated = False
        reason = "Normal"
        
        if pid_output > self.max_output:
            pid_output = self.max_output
            saturated = True
            reason = f"Saturação máxima ({self.max_output} U/h)"
        elif pid_output < 0:
            pid_output = 0  # Não há insulina negativa
            saturated = True
            reason = "Saturação mínima (0 U/h)"
        
        # Ajustar por IOB (insulina já ativa)
        effective_output = max(0, pid_output - (iob * 0.5))  # Fator empírico
        if effective_output != pid_output:
            reason += f" | Ajuste IOB: -{iob*0.5:.2f}"
        
        # Atualizar estado
        self.state.error = error
        self.state.previous_error = error
        self.state.last_output = effective_output
        self.state.last_time = current_time
        
        # Criar saída
        output = PIDOutput(
            dose=effective_output,
            p_component=p_component,
            i_component=i_component,
            d_component=d_component,
            saturated=saturated,
            reason=reason
        )
        
        # Salvar no histórico
        self._save_to_history(current_glucose, iob, output, dt)
        
        # Log detalhado para debugging
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"PID: G={current_glucose:.1f}, E={error:.1f}, "
                f"P={p_component:.3f}, I={i_component:.3f}, D={d_component:.3f}, "
                f"Out={effective_output:.3f}, IOB={iob:.2f}"
            )
        
        return output
    
    def _is_saturated(self) -> bool:
        """Verifica se a saída está saturada"""
        return (abs(self.state.last_output) >= self.max_output * 0.95)
    
    def _save_to_history(self, glucose: float, iob: float, 
                        output: PIDOutput, dt: float):
        """Salva dados no histórico para análise"""
        self.history.append({
            'timestamp': datetime.now(),
            'glucose': glucose,
            'iob': iob,
            'error': self.state.error,
            'output': output.dose,
            'p_component': output.p_component,
            'i_component': output.i_component,
            'd_component': output.d_component,
            'integral': self.state.integral,
            'dt': dt
        })
        
        # Manter apenas últimas 1000 entradas
        if len(self.history) > 1000:
            self.history.pop(0)
    
    def tune_parameters(self, kp: float = None, ki: float = None, 
                       kd: float = None):
        """
        Ajusta parâmetros PID em tempo real
        
        Args:
            kp: Novo ganho proporcional
            ki: Novo ganho integral  
            kd: Novo ganho derivativo
        """
        if kp is not None:
            self.kp = kp
        if ki is not None:
            self.ki = ki
        if kd is not None:
            self.kd = kd
            
        logger.info(f"Parâmetros PID atualizados: Kp={self.kp}, Ki={self.ki}, Kd={self.kd}")
    
    def reset(self):
        """Reseta estado interno do controlador"""
        self.state = PIDState()
        self.state.last_time = datetime.now()
        logger.info("Estado PID resetado")
    
    def get_performance_metrics(self, hours: int = 24) -> dict:
        """
        Calcula métricas de desempenho do controlador
        
        Args:
            hours: Período para análise em horas
            
        Returns:
            Dicionário com métricas calculadas
        """
        if not self.history:
            return {}
        
        # Filtrar dados recentes
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_data = [h for h in self.history if h['timestamp'] > cutoff_time]
        
        if not recent_data:
            return {}
        
        errors = [h['error'] for h in recent_data]
        outputs = [h['output'] for h in recent_data]
        
        metrics = {
            'mean_error': np.mean(errors),
            'rmse': np.sqrt(np.mean([e**2 for e in errors])),
            'mean_output': np.mean(outputs),
            'output_variability': np.std(outputs),
            'time_in_deadzone': sum(1 for e in errors if abs(e) < self.dead_zone) / len(errors),
            'saturation_rate': sum(1 for o in outputs if o >= self.max_output * 0.95) / len(outputs),
            'samples': len(recent_data)
        }
        
        return metrics
    
    def export_history(self, filename: str = None) -> str:
        """
        Exporta histórico para arquivo CSV
        
        Args:
            filename: Nome do arquivo (opcional)
            
        Returns:
            Caminho do arquivo criado
        """
        import pandas as pd
        
        if filename is None:
            filename = f"pid_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        df = pd.DataFrame(self.history)
        df.to_csv(filename, index=False)
        
        logger.info(f"Histórico PID exportado para {filename}")
        return filename

class AdaptivePIDController(PIDController):
    """
    Versão adaptativa do controlador PID que ajusta parâmetros automaticamente
    baseado no desempenho recente
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Parâmetros iniciais (backup)
        self.initial_kp = self.kp
        self.initial_ki = self.ki
        self.initial_kd = self.kd
        
        # Configuração adaptativa
        self.adaptation_enabled = True
        self.adaptation_interval = 60  # minutos
        self.last_adaptation = datetime.now()
        
        logger.info("Controlador PID adaptativo inicializado")
    
    def calculate(self, current_glucose: float, iob: float = 0.0,
                  dt: Optional[float] = None) -> PIDOutput:
        """
        Calcula saída com adaptação automática de parâmetros
        """
        # Executar PID normal
        output = super().calculate(current_glucose, iob, dt)
        
        # Verificar se é hora de adaptar
        if (self.adaptation_enabled and 
            (datetime.now() - self.last_adaptation).total_seconds() > self.adaptation_interval * 60):
            self._adapt_parameters()
            self.last_adaptation = datetime.now()
        
        return output
    
    def _adapt_parameters(self):
        """
        Adapta parâmetros PID baseado no desempenho recente
        """
        metrics = self.get_performance_metrics(hours=2)
        
        if not metrics or metrics['samples'] < 10:
            return
        
        # Ajustes baseados em métricas
        rmse = metrics['rmse']
        mean_error = metrics['mean_error']
        output_var = metrics['output_variability']
        
        # Estratégia simples de adaptação
        if rmse > 30:  # Erro alto
            if abs(mean_error) > 20:  # Erro sistemático
                self.kp *= 1.1  # Aumentar resposta proporcional
            if output_var > 2:  # Oscilação alta
                self.kd *= 1.2  # Aumentar amortecimento
        elif rmse < 15:  # Bom desempenho
            if output_var < 0.5:  # Resposta lenta
                self.kp *= 1.05  # Aumentar ligeiramente
        
        # Limitar ajustes para segurança
        self.kp = np.clip(self.kp, self.initial_kp * 0.5, self.initial_kp * 2.0)
        self.ki = np.clip(self.ki, self.initial_ki * 0.5, self.initial_ki * 2.0)
        self.kd = np.clip(self.kd, self.initial_kd * 0.5, self.initial_kd * 2.0)
        
        logger.info(f"Parâmetros PID adaptados: Kp={self.kp:.3f}, Ki={self.ki:.3f}, Kd={self.kd:.3f}")

# Funções utilitárias

def calculate_iob(insulin_history: List[dict], current_time: datetime = None) -> float:
    """
    Calcula Insulin on Board (insulina ativa no organismo)
    
    Args:
        insulin_history: Lista de doses de insulina com timestamps
        current_time: Tempo atual (padrão: agora)
        
    Returns:
        IOB em unidades de insulina
        
    Baseado no modelo exponencial de absorção de insulina:
    IOB(t) = dose * exp(-t/tau), onde tau ≈ 60 minutos para insulina rápida
    """
    if current_time is None:
        current_time = datetime.now()
    
    iob = 0.0
    tau = 60  # Constante de tempo em minutos (insulina rápida)
    
    for dose_record in insulin_history:
        dose_time = dose_record['timestamp']
        dose_amount = dose_record['dose']
        
        # Tempo decorrido em minutos
        elapsed_minutes = (current_time - dose_time).total_seconds() / 60.0
        
        if elapsed_minutes > 0 and elapsed_minutes < 300:  # 5 horas máximo
            # Modelo exponencial de absorção
            remaining_fraction = np.exp(-elapsed_minutes / tau)
            iob += dose_amount * remaining_fraction
    
    return iob

def estimate_glucose_impact(dose: float, isf: float, current_iob: float) -> float:
    """
    Estima impacto de dose de insulina na glicemia
    
    Args:
        dose: Dose de insulina em unidades
        isf: Fator de sensibilidade à insulina (mg/dL por unidade)
        current_iob: IOB atual em unidades
        
    Returns:
        Redução estimada de glicemia em mg/dL
    """
    # Ajustar por IOB existente (competição por receptores)
    effective_dose = dose * (1 - current_iob * 0.1)  # Fator empírico
    
    # Impacto estimado
    glucose_reduction = effective_dose * isf
    
    return glucose_reduction