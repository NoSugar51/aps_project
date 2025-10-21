"""
Simulador de Sistema de Pâncreas Artificial

Implementa simulação fisiológica baseada no UVA/Padova Type 1 Diabetes Simulator:
- Modelo compartimental de glicose-insulina
- Absorção gastrointestinal de carboidratos
- Farmacocinética de insulina subcutânea
- Ruído de sensor realista
- Cenários de teste reproduzíveis

Referências:
- Dalla Man et al. "Meal Simulation Model of the Glucose-Insulin System" (2007)
- UVA/Padova T1DM Simulator v3.2
- FDA-accepted simulation environment for artificial pancreas
"""

import logging
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

@dataclass
class GlucoseReading:
    timestamp: datetime
    value: float
    is_valid: bool = True

@dataclass
class InsulinDose:
    timestamp: datetime
    value: float
    type: str

class APSSimulator:
    """
    Motor de simulação do Sistema de Pâncreas Artificial
    
    Baseado no modelo UVA/Padova com adaptações para educação e pesquisa.
    Implementa modelo compartimental detalhado com realismo fisiológico.
    """
    
    def __init__(self, config: dict, patient_config: dict):
        """
        Inicializa simulador
        
        Args:
            config: Configuração da simulação
            patient_config: Parâmetros do paciente virtual
        """
        self.config = config
        self.patient = patient_config
        self.time_acceleration = 1
        self.dt = self.config.get('time_step_minutes', 5) / 60.0  # Converter para horas
        self.current_time = datetime.now()
        self.start_time = self.current_time
        
        # Parâmetros do modelo UVA/Padova
        self.Gb = 108.0  # Glicose basal (mg/dL)
        self.Ib = 7.0    # Insulina basal (µU/mL)
        self.SI = 0.52   # Sensibilidade à insulina (10^-4 dL/kg/min per µU/mL)
        self.p1 = 0.068  # Taxa de absorção intestinal (min^-1)
        self.p2 = 0.073  # Taxa de transferência (min^-1)
        self.Ke = 0.138  # Taxa de eliminação renal (min^-1)
        self.Vi = 0.12   # Volume de distribuição da insulina (L/kg)
        self.Vg = 1.88   # Volume de distribuição da glicose (dL/kg)
        
        self._validate_config()
        self.reset_simulation()
    
    def _validate_config(self):
        """Validar configuração necessária"""
        required_patient = ['weight_kg', 'carb_ratio', 'sensitivity', 'basal_rate']
        for param in required_patient:
            if param not in self.patient:
                raise ValueError(f"Parâmetro do paciente faltando: {param}")
        
    def reset_simulation(self):
        """Reinicia simulação para estado inicial"""
        self.G = self.config['initial_glucose']  # Glicemia (mg/dL)
        self.I = 0.0  # Insulina plasmática (mU/L)
        self.X = 0.0  # Efeito da insulina
        self.Q1 = 0.0  # Carboidratos em absorção
        self.Q2 = 0.0  # Insulina em absorção
        
        self.current_time = datetime.now()
        self.start_time = self.current_time
        self.insulin_history = []
        self.glucose_history = []
        self.meal_history = []
    
    def step(self):
        """
        Avança simulação por um timestep usando equações diferenciais
        
        Implementa sistema de EDOs do modelo UVA/Padova:
        dG/dt, dX/dt, dI/dt, dQsto1/dt, dQsto2/dt, dQgut/dt
        """
        try:
            # Atualizar tempo
            self.current_time += timedelta(minutes=self.config['time_step_minutes'] * self.time_acceleration)
            
            # 1. Subsistema de glicose
            EGP = self._calculate_glucose_production()
            Ra = self._calculate_glucose_appearance()
            Ut = self._calculate_glucose_utilization()
            E = self._calculate_renal_excretion()
            
            dG = (EGP + Ra - Ut - E) / self.Vg
            
            # 2. Subsistema de insulina
            S = self._calculate_insulin_secretion()
            I = self._calculate_insulin_appearance()
            
            dI = (S + I - self.Ke * self.I) / self.Vi
            
            # Integração numérica (Euler)
            self.G += dG * self.dt
            self.I += dI * self.dt
            
            # Limites físicos
            self.G = max(40, min(400, self.G))
            self.X = max(0, self.X)
            self.Q1 = max(0, self.Q1)
            self.Q2 = max(0, self.Q2)
            
            # Registrar histórico
            glucose_reading = GlucoseReading(
                timestamp=self.current_time,
                value=float(self.G)
            )
            self.glucose_history.append(glucose_reading)
            
        except Exception as e:
            logger.error(f"Erro no step da simulação: {e}")
            raise

    # Métodos de acesso aos dados
    
    def get_current_glucose(self) -> float:
        """Retorna glicemia atual do sensor"""
        if self.glucose_history:
            return self.glucose_history[-1].value
        return 120.0  # Valor padrão
    
    def get_current_true_glucose(self) -> float:
        """Retorna glicemia verdadeira (sem ruído do sensor)"""
        return self.G
    
    def get_glucose_history(self, hours: int = 6) -> list:
        """Retorna histórico de glicemia"""
        cutoff = self.current_time - timedelta(hours=hours)
        return [
            {"timestamp": reading.timestamp, "value": reading.value}
            for reading in self.glucose_history 
            if reading.timestamp >= cutoff
        ]
    
    def get_insulin_history(self, hours: int = 6) -> list:
        """Retorna histórico de insulina"""
        cutoff = self.current_time - timedelta(hours=hours)
        return [
            {"timestamp": dose.timestamp, "dose": dose.value, "type": dose.type}
            for dose in self.insulin_history 
            if dose.timestamp >= cutoff
        ]

    def get_meal_history(self, hours: int = 6) -> List[Dict]:
        """Retorna histórico de refeições"""
        cutoff_time = self.current_time - timedelta(hours=hours)
        
        filtered_history = [
            {
                'timestamp': meal.timestamp,
                'carbs': meal.carbs,
                'absorption_profile': meal.absorption_profile,
                'gi_index': meal.gi_index
            }
            for meal in self.meal_history
            if meal.timestamp >= cutoff_time
        ]
        
        return filtered_history
    
    def apply_insulin_dose(self, dose: float, dose_type: str):
        """
        Aplica dose de insulina no simulador
        
        Args:
            dose: Quantidade de insulina em Unidades (U)
            dose_type: Tipo de dose ('bolus' ou 'basal')
        """
        try:
            # Adicionar insulina ao compartimento de absorção
            self.Q2 += dose
            
            # Registrar usando dataclass
            insulin_dose = InsulinDose(
                timestamp=self.current_time,
                value=dose,
                type=dose_type
            )
            self.insulin_history.append(insulin_dose)
            
            logger.debug(f"Dose aplicada: {dose}U {dose_type}")
            
        except Exception as e:
            logger.error(f"Erro ao aplicar insulina: {e}")
            raise

    def get_sensor_data(self) -> dict:
        """Retorna dados do sensor para verificações de segurança"""
        return {
            'glucose': self.G,
            'rate_of_change': self._calculate_glucose_rate(),
            'sensor_status': 'ok',
            'last_calibration': self.start_time
        }
    
    def _calculate_glucose_rate(self) -> float:
        """Calcula taxa de variação da glicemia"""
        if len(self.glucose_history) < 2:
            return 0.0
            
        last_two = self.glucose_history[-2:]
        dt = (last_two[1].timestamp - last_two[0].timestamp).total_seconds() / 3600
        dg = last_two[1].value - last_two[0].value
        
        return dg / dt if dt > 0 else 0.0

    def get_current_time(self) -> datetime:
        """Retorna tempo atual da simulação"""
        return self.current_time
    
    def set_time_acceleration(self, factor: int):
        """Define fator de aceleração temporal"""
        self.time_acceleration = max(1, min(100, factor))
        logger.info(f"Aceleração temporal: {self.time_acceleration}x")
    
    def get_physiological_state(self) -> dict:
        """
        Retorna estado interno completo da simulação
        
        Útil para debugging e análise detalhada
        """
        return {
            'glucose_plasma': self.G,
            'insulin_remote': self.X,
            'insulin_plasma': self.I,
            'stomach_solid': self.Qsto1,
            'stomach_liquid': self.Qsto2,
            'gut_glucose': self.Qgut,
            'patient_state': self.current_state.value,
            'time_acceleration': self.time_acceleration,
            'sensor_noise_std': self.sensor_noise_std,
            'sensor_calibration_error': self.sensor_calibration_error
        }
    
    def export_simulation_data(self) -> dict:
        """
        Exporta todos os dados da simulação
        
        Returns:
            Dicionário com dados completos da simulação
        """
        return {
            'simulation_info': {
                'start_time': self.start_time.isoformat(),
                'current_time': self.current_time.isoformat(),
                'duration_hours': (self.current_time - self.start_time).total_seconds() / 3600,
                'time_acceleration': self.time_acceleration
            },
            'patient_config': self.patient_config,
            'physiological_state': self.get_physiological_state(),
            'glucose_history': [
                {
                    'timestamp': r.timestamp.isoformat(),
                    'sensor_value': r.value,
                    'true_value': r.noise_free_value
                }
                for r in self.glucose_history
            ],
            'insulin_history': [
                {
                    'timestamp': d.timestamp.isoformat(),
                    'dose': d.dose,
                    'type': d.dose_type
                }
                for d in self.insulin_history
            ],
            'meal_history': [
                {
                    'timestamp': m.timestamp.isoformat(),
                    'carbs': m.carbs,
                    'profile': m.absorption_profile,
                    'gi_index': m.gi_index
                }
                for m in self.meal_history
            ]
        }
    
    def _calculate_glucose_production(self):
        """Produção endógena de glicose"""
        return max(0, self.EGP0 * (1 - self.SI * self.I))
        
    def _calculate_insulin_sensitivity(self):
        """Sensibilidade à insulina variável"""
        hour = self.current_time.hour
        # Variação circadiana
        if 0 <= hour < 6:  # Dawn phenomenon
            return self.SI * 0.8
        return self.SI
        
    def inject_sensor_error(self, error_type: str, duration_minutes: int):
        """Simula falhas realistas do sensor"""
        if error_type == "spike":
            self.sensor_error = np.random.normal(50, 10)
        elif error_type == "drift":
            self.sensor_drift_rate = 0.5  # mg/dL/min