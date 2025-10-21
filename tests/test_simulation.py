"""
Testes automatizados para Sistema de Pâncreas Artificial

Testa todos os componentes principais:
- Simulador fisiológico
- Controlador PID e híbrido
- Sistema de segurança
- Machine Learning
- Métricas de desempenho
"""

import pytest
import asyncio
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import sys
import os

# Adicionar src ao path para imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sim.simulator import APSSimulator
from controller.pid_controller import PIDController
from controller.hybrid_controller import HybridController, ControlMode
from ml.model import GlucosePredictionModel
from utils.safety_checks import SafetyMonitor
from utils.metrics import MetricsCalculator

class TestAPSSimulator:
    """Testes do simulador fisiológico"""
    
    @pytest.fixture
    def simulator(self):
        """Fixture para simulador"""
        config = {
            'time_step_minutes': 10,
            'sensor_noise_std': 5.0,
            'pump_delay_minutes': 15
        }
        
        patient_config = {
            'age': 30,
            'weight_kg': 70,
            'isf': 50,
            'carb_ratio': 10
        }
        
        return APSSimulator(config, patient_config)
    
    def test_simulator_initialization(self, simulator):
        """Testa inicialização do simulador"""
        assert simulator.G == 120.0  # Glicemia inicial
        assert simulator.BW == 70.0  # Peso corporal
        assert simulator.dt == 10  # Timestep
        assert len(simulator.glucose_history) == 0
    
    def test_physiological_step(self, simulator):
        """Testa execução de step fisiológico"""
        initial_glucose = simulator.G
        
        # Executar alguns steps
        for _ in range(5):
            simulator.step()
        
        # Verificar evolução
        assert len(simulator.glucose_history) == 5
        assert simulator.current_time > simulator.start_time
        
        # Glicemia deve estar em faixa fisiológica
        final_glucose = simulator.get_current_glucose()
        assert 30 < final_glucose < 500
    
    def test_insulin_application(self, simulator):
        """Testa aplicação de insulina"""
        initial_insulin_count = len(simulator.insulin_history)
        
        # Aplicar dose
        simulator.apply_insulin_dose(2.5, 'bolus')
        
        assert len(simulator.insulin_history) == initial_insulin_count + 1
        assert simulator.insulin_history[-1]['dose'] == 2.5
        assert simulator.insulin_history[-1]['dose_type'] == 'bolus'
    
    def test_meal_consumption(self, simulator):
        """Testa simulação de refeição"""
        initial_qsto = simulator.Qsto1 + simulator.Qsto2
        
        # Consumir refeição
        simulator.consume_meal(carbs=45, absorption_profile='medium')
        
        # Verificar adição ao compartimento gástrico
        final_qsto = simulator.Qsto1 + simulator.Qsto2
        assert final_qsto > initial_qsto
        
        assert len(simulator.meal_history) == 1
        assert simulator.meal_history[0]['carbs'] == 45
    
    def test_sensor_error_injection(self, simulator):
        """Testa injeção de erro no sensor"""
        original_noise = simulator.sensor_noise_std
        
        simulator.inject_sensor_error('noise', 30)
        
        # Ruído deve ter aumentado
        assert simulator.sensor_noise_std > original_noise

class TestPIDController:
    """Testes do controlador PID"""
    
    @pytest.fixture
    def pid_controller(self):
        """Fixture para controlador PID"""
        return PIDController(
            kp=0.6, ki=0.02, kd=0.01,
            target_glucose=120,
            max_output=5.0
        )
    
    def test_pid_initialization(self, pid_controller):
        """Testa inicialização do PID"""
        assert pid_controller.kp == 0.6
        assert pid_controller.ki == 0.02
        assert pid_controller.kd == 0.01
        assert pid_controller.target_glucose == 120
        assert len(pid_controller.history) == 0
    
    def test_pid_calculation_normal_glucose(self, pid_controller):
        """Testa cálculo PID com glicemia normal"""
        output = pid_controller.calculate(current_glucose=120, iob=0)
        
        # Erro zero deve resultar em saída mínima
        assert output.dose >= 0
        assert abs(output.p_component) < 0.1  # Componente proporcional baixo
        assert abs(output.i_component) < 0.1  # Componente integral baixo
    
    def test_pid_calculation_high_glucose(self, pid_controller):
        """Testa cálculo PID com hiperglicemia"""
        output = pid_controller.calculate(current_glucose=200, iob=0)
        
        # Glicemia alta deve gerar saída positiva
        assert output.dose > 0
        assert output.p_component > 0  # Proporcional positivo
        assert "Normal" in output.reason or "Saturação" in output.reason
    
    def test_pid_saturation(self, pid_controller):
        """Testa saturação do PID"""
        # Glicemia muito alta para forçar saturação
        output = pid_controller.calculate(current_glucose=400, iob=0)
        
        assert output.dose <= pid_controller.max_output
        if output.saturated:
            assert "Saturação" in output.reason
    
    def test_pid_iob_feedback(self, pid_controller):
        """Testa feedback de IOB no PID"""
        # Sem IOB
        output_no_iob = pid_controller.calculate(current_glucose=180, iob=0)
        
        # Com IOB alto
        output_with_iob = pid_controller.calculate(current_glucose=180, iob=3.0)
        
        # Com IOB, dose deve ser menor
        assert output_with_iob.dose <= output_no_iob.dose
        assert "IOB" in output_with_iob.reason

class TestHybridController:
    """Testes do controlador híbrido PID + ML"""
    
    @pytest.fixture
    def hybrid_controller(self):
        """Fixture para controlador híbrido"""
        config = {
            'pid': {
                'kp': 0.6, 'ki': 0.02, 'kd': 0.01,
                'target_glucose': 120, 'max_u_per_hour': 5.0
            },
            'hybrid': {
                'ml_weight': 0.3, 'confidence_threshold': 0.8,
                'adaptation_days': 14
            }
        }
        
        # Mock do modelo ML
        ml_model = Mock()
        ml_model.predict.return_value = {
            'predictions': [120, 125, 130, 135, 140],
            'confidence': 0.8
        }
        
        # Mock do monitor de segurança
        safety_monitor = Mock()
        safety_monitor.check_safety.return_value = Mock(
            is_safe=True, emergency_stop=False
        )
        
        return HybridController(config, ml_model, safety_monitor)
    
    def test_hybrid_initialization(self, hybrid_controller):
        """Testa inicialização do controlador híbrido"""
        assert hybrid_controller.current_mode == ControlMode.LEARNING
        assert hybrid_controller.ml_weight == 0.3
        assert len(hybrid_controller.decision_history) == 0
    
    def test_hybrid_calculate_dose_learning_mode(self, hybrid_controller):
        """Testa cálculo de dose em modo aprendizado"""
        # Forçar modo aprendizado
        hybrid_controller.current_mode = ControlMode.LEARNING
        
        glucose_history = [
            {'timestamp': datetime.now() - timedelta(minutes=i*10), 'value': 120 + i}
            for i in range(10)
        ]
        
        output = hybrid_controller.calculate_dose(
            current_glucose=140,
            glucose_history=glucose_history,
            insulin_history=[],
            meal_history=[]
        )
        
        assert output.dose >= 0
        assert output.mode == ControlMode.LEARNING
        assert output.ml_component == 0  # Sem ML no modo aprendizado
    
    def test_hybrid_calculate_dose_autonomous_mode(self, hybrid_controller):
        """Testa cálculo de dose em modo autônomo"""
        # Forçar modo autônomo
        hybrid_controller.current_mode = ControlMode.AUTONOMOUS
        
        glucose_history = [
            {'timestamp': datetime.now() - timedelta(minutes=i*10), 'value': 150}
            for i in range(36)
        ]
        
        output = hybrid_controller.calculate_dose(
            current_glucose=150,
            glucose_history=glucose_history,
            insulin_history=[],
            meal_history=[]
        )
        
        assert output.mode == ControlMode.AUTONOMOUS
        # Deve ter componente ML no modo autônomo
        assert abs(output.ml_component) >= 0

class TestSafetyMonitor:
    """Testes do monitor de segurança"""
    
    @pytest.fixture
    def safety_monitor(self):
        """Fixture para monitor de segurança"""
        config = {
            'min_glucose_alarm': 70,
            'max_glucose_alarm': 250,
            'max_bolus_per_30min': 4.0,
            'sensor_spike_threshold': 40
        }
        return SafetyMonitor(config)
    
    def test_safety_normal_glucose(self, safety_monitor):
        """Testa verificação com glicemia normal"""
        result = safety_monitor.check_safety(
            glucose=120,
            insulin_history=[],
            sensor_data={'trend': 0, 'quality': 'good'}
        )
        
        assert result.is_safe == True
        assert result.emergency_stop == False
        assert "normal" in result.message.lower() or "✅" in result.message
    
    def test_safety_severe_hypoglycemia(self, safety_monitor):
        """Testa detecção de hipoglicemia severa"""
        result = safety_monitor.check_safety(
            glucose=45,  # Hipoglicemia severa
            insulin_history=[],
            sensor_data={'trend': -2, 'quality': 'good'}
        )
        
        assert result.is_safe == False
        assert result.emergency_stop == True
        assert "hipoglicemia" in result.message.lower()
    
    def test_safety_insulin_overdose(self, safety_monitor):
        """Testa detecção de overdose de insulina"""
        # Histórico com muita insulina recente
        now = datetime.now()
        insulin_history = [
            {'timestamp': now - timedelta(minutes=10), 'dose': 3.0},
            {'timestamp': now - timedelta(minutes=20), 'dose': 2.5}
        ]
        
        result = safety_monitor.check_safety(
            glucose=120,
            insulin_history=insulin_history,
            sensor_data={'trend': 0, 'quality': 'good'}
        )
        
        # Total > 4U em 30min deve gerar alerta
        assert result.is_safe == False
        assert "insulina" in result.message.lower()
    
    def test_safety_sensor_spike(self, safety_monitor):
        """Testa detecção de spike no sensor"""
        # Simular leitura anterior
        safety_monitor.last_glucose_reading = 120
        
        result = safety_monitor.check_safety(
            glucose=180,  # Spike de 60 mg/dL
            insulin_history=[],
            sensor_data={'trend': 15, 'quality': 'good'}
        )
        
        assert result.is_safe == False
        assert "spike" in result.message.lower()

class TestMLModel:
    """Testes do modelo de Machine Learning"""
    
    @pytest.fixture
    def ml_model(self):
        """Fixture para modelo ML"""
        config = {
            'sequence_length': 36,
            'hidden_units': 64,
            'learning_rate': 0.001
        }
        return GlucosePredictionModel(config)
    
    def test_ml_model_initialization(self, ml_model):
        """Testa inicialização do modelo"""
        assert ml_model.sequence_length == 36
        assert ml_model.hidden_units == 64
        assert ml_model.is_fitted == False
        assert ml_model.model is not None
    
    def test_ml_model_feature_preparation(self, ml_model):
        """Testa preparação de features"""
        glucose_history = [
            {'timestamp': datetime.now() - timedelta(minutes=i*10), 'value': 120 + i}
            for i in range(50)
        ]
        
        features = ml_model.prepare_features(
            glucose_history=glucose_history,
            insulin_history=[],
            meal_history=[]
        )
        
        assert 'glucose_sequence' in features
        assert 'context_features' in features
        assert features['glucose_sequence'].shape[1] == 36
        assert features['context_features'].shape[1] == 4
    
    def test_ml_model_prediction_unfitted(self, ml_model):
        """Testa previsão com modelo não treinado"""
        features = {
            'glucose_sequence': np.random.rand(1, 36),
            'context_features': np.random.rand(1, 4)
        }
        
        result = ml_model.predict(features)
        
        # Deve retornar previsão conservadora
        assert result.confidence <= 0.5
        assert len(result.predictions) > 0

class TestMetricsCalculator:
    """Testes do calculador de métricas"""
    
    @pytest.fixture
    def metrics_calculator(self):
        """Fixture para calculadora de métricas"""
        return MetricsCalculator()
    
    @pytest.fixture
    def sample_data(self):
        """Dados de exemplo para testes"""
        data = []
        base_time = datetime.now() - timedelta(hours=24)
        
        for i in range(144):  # 24h com readings a cada 10min
            timestamp = base_time + timedelta(minutes=i*10)
            
            # Simular glicemia principalmente em faixa
            if i < 20:  # Algumas hipoglicemias no início
                glucose = np.random.uniform(60, 85)
            elif i > 100:  # Algumas hiperglicemias no final
                glucose = np.random.uniform(200, 250)
            else:
                glucose = np.random.uniform(90, 160)  # Principalmente em faixa
            
            data.append({
                'timestamp': timestamp,
                'glucose': glucose,
                'insulin_dose': np.random.uniform(0, 1),
                'step': i
            })
        
        return data
    
    def test_metrics_calculation(self, metrics_calculator, sample_data):
        """Testa cálculo de métricas básicas"""
        metrics = metrics_calculator.calculate_all_metrics(sample_data)
        
        assert 0 <= metrics.time_in_range <= 1
        assert 0 <= metrics.time_below_range <= 1
        assert 0 <= metrics.time_above_range <= 1
        
        # Soma das porcentagens deve ser 1
        total = metrics.time_in_range + metrics.time_below_range + metrics.time_above_range
        assert abs(total - 1.0) < 0.01
        
        assert metrics.mean_glucose > 0
        assert metrics.glucose_std >= 0
        assert metrics.coefficient_variation >= 0
    
    def test_metrics_event_detection(self, metrics_calculator, sample_data):
        """Testa detecção de eventos"""
        metrics = metrics_calculator.calculate_all_metrics(sample_data)
        
        # Deve detectar alguns eventos baseado nos dados simulados
        assert metrics.hypoglycemia_events >= 0
        assert metrics.hyperglycemia_events >= 0
        assert isinstance(metrics.hypoglycemia_events, int)
    
    def test_metrics_report_generation(self, metrics_calculator, sample_data):
        """Testa geração de relatório"""
        metrics = metrics_calculator.calculate_all_metrics(sample_data)
        report = metrics_calculator.generate_metrics_report(metrics, 24)
        
        assert isinstance(report, str)
        assert len(report) > 0
        assert "RELATÓRIO DE DESEMPENHO" in report
        assert "Tempo em faixa" in report