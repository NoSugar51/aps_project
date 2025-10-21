"""
Modelo de Machine Learning para Previsão Glicêmica

Implementa rede neural LSTM para:
- Previsão de glicemia futura (10, 20, 30, 60, 120 minutos)
- Estimativa de confiança das previsões
- Treinamento incremental online
- Detecção de padrões personalizados

Baseado em:
- LSTM for Time Series Prediction in Healthcare
- Personalized Glucose Prediction Models
- Online Learning for Continuous Glucose Monitoring
"""

import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import pickle
import os
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)

# Configurar TensorFlow para usar CPU (mais estável em produção)
tf.config.set_visible_devices([], 'GPU')

@dataclass
class PredictionResult:
    """Resultado da previsão do modelo"""
    predictions: List[float]  # Glicemia prevista em diferentes horizontes
    confidence: float  # Confiança geral (0-1)
    uncertainty: List[float]  # Incerteza por horizonte
    features_used: int  # Número de features utilizadas

class GlucosePredictionModel:
    """
    Modelo LSTM para previsão de glicemia com aprendizado incremental
    
    Arquitetura:
    - Input: Sequência de 36 timesteps (6h com intervalos de 10min)
    - Features: glicose, insulina, carboidratos, tempo, contexto
    - LSTM layers: 64 -> 32 unidades
    - Output: 5 previsões (10, 20, 30, 60, 120 min)
    """
    
    def __init__(self, config: dict):
        """
        Inicializa modelo LSTM
        
        Args:
            config: Configuração do modelo ML
        """
        self.config = config
        self.sequence_length = config.get('sequence_length', 36)  # 6h em steps de 10min
        self.hidden_units = config.get('hidden_units', 64)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.batch_size = config.get('batch_size', 32)
        
        # Horizontes de previsão (em steps de 10min)
        self.prediction_horizons = [1, 2, 3, 6, 12]  # 10, 20, 30, 60, 120 min
        self.n_outputs = len(self.prediction_horizons)
        
        # Preprocessamento
        self.scaler_glucose = StandardScaler()
        self.scaler_features = StandardScaler()
        self.is_fitted = False
        
        # Modelo e histórico
        self.model = None
        self.training_history = []
        self.prediction_errors = []
        
        # Métricas de confiança
        self.confidence_threshold = 0.8
        self.recent_mae = []
        
        logger.info("Modelo LSTM inicializado")
        self._build_model()
    
    def _build_model(self):
        """Constrói arquitetura do modelo LSTM"""
        try:
            # Definir entrada
            # Features: [glucose_sequence(36), insulin_active(1), carbs_recent(1), 
            #           hour_of_day(1), day_of_week(1)]
            n_features = 36 + 4  # 36 glucose + 4 context features
            
            # Input layers
            sequence_input = keras.Input(shape=(self.sequence_length,), name='glucose_sequence')
            context_input = keras.Input(shape=(4,), name='context_features')
            
            # LSTM para sequência de glicose
            x = layers.Reshape((self.sequence_length, 1))(sequence_input)
            x = layers.LSTM(self.hidden_units, return_sequences=True, 
                           dropout=0.2, recurrent_dropout=0.2)(x)
            x = layers.LSTM(32, dropout=0.2)(x)
            
            # Combinar com features de contexto
            combined = layers.concatenate([x, context_input])
            
            # Dense layers para previsão
            dense = layers.Dense(64, activation='relu')(combined)
            dense = layers.Dropout(0.3)(dense)
            dense = layers.Dense(32, activation='relu')(dense)
            
            # Output: múltiplas previsões
            predictions = layers.Dense(self.n_outputs, activation='linear', name='predictions')(dense)
            
            # Output adicional: estimativa de incerteza
            uncertainty = layers.Dense(self.n_outputs, activation='sigmoid', name='uncertainty')(dense)
            
            # Criar modelo
            self.model = keras.Model(
                inputs=[sequence_input, context_input],
                outputs=[predictions, uncertainty]
            )
            
            # Compilar com loss personalizada
            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
                loss={
                    'predictions': 'mse',
                    'uncertainty': 'binary_crossentropy'
                },
                loss_weights={
                    'predictions': 1.0,
                    'uncertainty': 0.1  # Peso menor para incerteza
                },
                metrics={
                    'predictions': ['mae', 'mse'],
                    'uncertainty': ['binary_accuracy']
                }
            )
            
            logger.info(f"Modelo LSTM construído: {self.model.count_params()} parâmetros")
            
        except Exception as e:
            logger.error(f"Erro ao construir modelo: {e}")
            raise
    
    def prepare_features(self, glucose_history: List[dict], insulin_history: List[dict],
                        meal_history: List[dict], current_time: datetime = None) -> np.ndarray:
        """
        Prepara features para previsão
        
        Args:
            glucose_history: Histórico de glicemia
            insulin_history: Histórico de insulina
            meal_history: Histórico de refeições
            current_time: Tempo atual (padrão: agora)
            
        Returns:
            Array com features preparadas
        """
        if current_time is None:
            current_time = current_time or datetime.now()
        
        # 1. Sequência de glicose (últimas 6h)
        glucose_sequence = self._extract_glucose_sequence(glucose_history, current_time)
        
        # 2. Features de contexto
        context_features = self._extract_context_features(
            insulin_history, meal_history, current_time
        )
        
        return {
            'glucose_sequence': glucose_sequence.reshape(1, -1),
            'context_features': context_features.reshape(1, -1)
        }
    
    def _extract_glucose_sequence(self, glucose_history: List[dict], 
                                current_time: datetime) -> np.ndarray:
        """Extrai sequência de glicose das últimas 6h"""
        # Criar array temporal
        sequence = np.full(self.sequence_length, 120.0)  # Valor padrão
        
        if not glucose_history:
            return sequence
        
        # Filtrar últimas 6h e ordenar por tempo
        cutoff_time = current_time - timedelta(hours=6)
        recent_glucose = [
            g for g in glucose_history 
            if g['timestamp'] >= cutoff_time
        ]
        recent_glucose.sort(key=lambda x: x['timestamp'])
        
        # Preencher sequência (interpolação simples)
        if recent_glucose:
            # Mapear timestamps para índices da sequência
            start_time = cutoff_time
            for i, glucose_point in enumerate(recent_glucose):
                elapsed_minutes = (glucose_point['timestamp'] - start_time).total_seconds() / 60
                sequence_index = int(elapsed_minutes / 10)  # Steps de 10 min
                
                if 0 <= sequence_index < self.sequence_length:
                    sequence[sequence_index] = glucose_point['value']
            
            # Interpolação linear para pontos faltantes
            mask = sequence != 120.0  # Pontos não preenchidos
            if np.any(mask):
                indices = np.arange(len(sequence))
                sequence = np.interp(indices, indices[mask], sequence[mask])
        
        return sequence
    
    def _extract_context_features(self, insulin_history: List[dict], 
                                meal_history: List[dict], current_time: datetime) -> np.ndarray:
        """Extrai features de contexto"""
        features = np.zeros(4)
        
        # 1. Insulina ativa (IOB)
        features[0] = self._calculate_iob(insulin_history, current_time)
        
        # 2. Carboidratos recentes (últimas 4h)
        features[1] = self._calculate_recent_carbs(meal_history, current_time)
        
        # 3. Hora do dia (normalizada)
        features[2] = current_time.hour / 24.0
        
        # 4. Dia da semana (normalizado)
        features[3] = current_time.weekday() / 6.0
        
        return features
    
    def _calculate_iob(self, insulin_history: List[dict], current_time: datetime) -> float:
        """Calcula insulina ativa simplificada"""
        iob = 0.0
        tau = 60  # Constante de tempo em minutos
        
        for dose in insulin_history:
            elapsed_minutes = (current_time - dose['timestamp']).total_seconds() / 60
            if 0 < elapsed_minutes < 300:  # Máximo 5h
                remaining = np.exp(-elapsed_minutes / tau)
                iob += dose['dose'] * remaining
        
        return min(iob, 20.0)  # Limitar para normalização
    
    def _calculate_recent_carbs(self, meal_history: List[dict], current_time: datetime) -> float:
        """Calcula carboidratos nas últimas 4h"""
        cutoff_time = current_time - timedelta(hours=4)
        
        total_carbs = 0
        for meal in meal_history:
            if meal['timestamp'] >= cutoff_time:
                total_carbs += meal.get('carbs', 0)
        
        return min(total_carbs / 100.0, 2.0)  # Normalizar
    
    def predict(self, features: Dict[str, np.ndarray]) -> PredictionResult:
        """
        Faz previsão de glicemia futura
        
        Args:
            features: Features preparadas
            
        Returns:
            PredictionResult com previsões e confiança
        """
        try:
            if self.model is None or not self.is_fitted:
                # Modelo não treinado - retornar previsão conservadora
                return self._conservative_prediction()
            
            # Fazer previsão
            pred_output, uncertainty_output = self.model.predict([
                features['glucose_sequence'],
                features['context_features']
            ], verbose=0)
            
            # Extrair previsões e incertezas
            predictions = pred_output[0].tolist()
            uncertainties = uncertainty_output[0].tolist()
            
            # Calcular confiança geral
            confidence = self._calculate_confidence(predictions, uncertainties)
            
            # Aplicar limites fisiológicos
            predictions = [max(40, min(400, p)) for p in predictions]
            
            return PredictionResult(
                predictions=predictions,
                confidence=confidence,
                uncertainty=uncertainties,
                features_used=features['glucose_sequence'].shape[1] + features['context_features'].shape[1]
            )
            
        except Exception as e:
            logger.error(f"Erro na previsão: {e}")
            return self._conservative_prediction()
    
    def _conservative_prediction(self) -> PredictionResult:
        """Previsão conservadora quando modelo não disponível"""
        # Assumir glicemia estável em 120 mg/dL
        return PredictionResult(
            predictions=[120.0] * self.n_outputs,
            confidence=0.3,  # Baixa confiança
            uncertainty=[0.8] * self.n_outputs,  # Alta incerteza
            features_used=0
        )
    
    def _calculate_confidence(self, predictions: List[float], uncertainties: List[float]) -> float:
        """
        Calcula confiança da previsão baseado em múltiplos fatores
        
        Args:
            predictions: Previsões do modelo
            uncertainties: Incertezas estimadas
            
        Returns:
            Confiança entre 0 e 1
        """
        # Confiança base: inverso da incerteza média
        base_confidence = 1.0 - np.mean(uncertainties)
        
        # Ajustar por consistência das previsões
        pred_std = np.std(predictions)
        if pred_std > 50:  # Previsões inconsistentes
            base_confidence *= 0.7
        
        # Ajustar por histórico de erros recentes
        if self.recent_mae:
            recent_error = np.mean(self.recent_mae[-20:])  # Últimos 20 erros
            if recent_error > 30:  # MAE alto
               base_confidence *= 0.5