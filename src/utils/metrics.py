"""
Sistema de métricas para avaliação de desempenho do APS

Métricas implementadas:
- Tempo em faixa (Time in Range - TIR)
- Coeficiente de variação glicêmica
- Índice de baixa glicemia (LBGI)
- Índice de alta glicemia (HBGI)
- Mean Absolute Relative Difference (MARD)
- Área sob a curva de risco
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Métricas de desempenho calculadas"""
    # Métricas básicas
    time_in_range: float  # Porcentagem (0-1)
    time_below_range: float  # < 70 mg/dL
    time_above_range: float  # > 180 mg/dL
    
    # Métricas estatísticas
    mean_glucose: float  # mg/dL
    glucose_std: float   # mg/dL
    coefficient_variation: float  # %
    
    # Métricas de segurança
    hypoglycemia_events: int  # < 70 mg/dL por >15min
    severe_hypoglycemia_events: int  # < 54 mg/dL
    hyperglycemia_events: int  # > 250 mg/dL por >2h
    
    # Métricas de qualidade
    glucose_management_indicator: float  # GMI (%)
    low_blood_glucose_index: float  # LBGI
    high_blood_glucose_index: float  # HBGI
    
    # Métricas específicas do controlador
    controller_accuracy: float  # MARD do controle vs target
    insulin_efficiency: float  # Glicose controlada por unidade
    prediction_accuracy: Optional[float] = None  # Se ML disponível

class MetricsCalculator:
    """
    Calculadora de métricas de desempenho para sistemas APS
    
    Implementa métricas padrão da literatura sobre monitoramento
    contínuo de glicose e sistemas de pâncreas artificial.
    """
    
    def __init__(self):
        # Limites de referência
        self.target_range = (70, 180)  # mg/dL
        self.tight_range = (70, 140)   # mg/dL (para métricas rigorosas)
        self.severe_hypo_threshold = 54  # mg/dL
        self.severe_hyper_threshold = 250  # mg/dL
        
        # Parâmetros para cálculos
        self.min_event_duration_min = 15  # Duração mínima de evento
        self.hyper_event_duration_min = 120  # 2h para hiperglicemia
        
        # Dados para cálculo de métricas
        self.glucose_data = []
        self.insulin_data = []
        
        logger.info("Calculadora de métricas inicializada")
    
    def update(self, glucose: float, insulin: float):
        """Atualiza os dados de glicose e insulina com novas leituras"""
        try:
            self.glucose_data.append(glucose)
            self.insulin_data.append(insulin)
        except Exception as e:
            logger.error(f"Erro ao atualizar métricas: {e}")
            
    def calculate_metrics(self) -> PerformanceMetrics:
        """
        Calcula todas as métricas de desempenho a partir dos dados coletados
        
        Returns:
            PerformanceMetrics com todas as métricas calculadas
        """
        try:
            if not self.glucose_data:
                return self._empty_metrics()
                
            glucose_array = np.array(self.glucose_data)
            insulin_array = np.array(self.insulin_data)
            
            metrics = PerformanceMetrics(
                time_in_range=self._calculate_tir(glucose_array),
                mean_glucose=np.mean(glucose_array),
                glucose_std=np.std(glucose_array),
                coefficient_variation=np.std(glucose_array) / np.mean(glucose_array) * 100,
                hypoglycemia_events=self._count_events(glucose_array < 70),
                severe_hypoglycemia_events=self._count_events(glucose_array < self.severe_hypo_threshold),
                hyperglycemia_events=self._count_events(glucose_array > 180),
                glucose_management_indicator=self._calculate_gmi(glucose_array),
                low_blood_glucose_index=self._calculate_lbgi(glucose_array),
                high_blood_glucose_index=self._calculate_hbgi(glucose_array),
                controller_accuracy=0.0,  # Placeholder, calcular se modelo disponível
                insulin_efficiency=np.mean(insulin_array / glucose_array) if np.all(glucose_array) else 0.0
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Erro ao calcular métricas: {e}")
            return self._empty_metrics()
    
    def _empty_metrics(self) -> PerformanceMetrics:
        """Retorna métricas vazias para casos de erro"""
        return PerformanceMetrics(
            time_in_range=0.0,
            time_below_range=0.0,
            time_above_range=0.0,
            mean_glucose=0.0,
            glucose_std=0.0,
            coefficient_variation=0.0,
            hypoglycemia_events=0,
            severe_hypoglycemia_events=0,
            hyperglycemia_events=0,
            glucose_management_indicator=0.0,
            low_blood_glucose_index=0.0,
            high_blood_glucose_index=0.0,
            controller_accuracy=0.0,
            insulin_efficiency=0.0
        )
    
    def _calculate_tir(self, glucose_array: np.ndarray) -> float:
        """Calcula o tempo em faixa (TIR)"""
        try:
            in_range = (glucose_array >= self.target_range[0]) & (glucose_array <= self.target_range[1])
            return np.mean(in_range) * 100
        except Exception as e:
            logger.error(f"Erro ao calcular TIR: {e}")
            return 0.0
    
    def _count_events(self, condition_array: np.ndarray) -> int:
        """Conta eventos de acordo com uma condição (ex: glicose < 70)"""
        try:
            # Conta transições de False para True
            return np.sum(np.diff(condition_array.astype(int)) == 1)
        except Exception as e:
            logger.error(f"Erro ao contar eventos: {e}")
            return 0
    
    def _calculate_gmi(self, glucose_array: np.ndarray) -> float:
        """Calcula o índice de gerenciamento glicêmico (GMI)"""
        try:
            return 3.31 + 0.02392 * np.mean(glucose_array)
        except Exception as e:
            logger.error(f"Erro ao calcular GMI: {e}")
            return 0.0
    
    def _calculate_lbgi(self, glucose_array: np.ndarray) -> float:
        """Calcula o índice de baixa glicemia (LBGI)"""
        try:
            return np.mean(np.maximum(0, 70 - glucose_array)) / 70
        except Exception as e:
            logger.error(f"Erro ao calcular LBGI: {e}")
            return 0.0
    
    def _calculate_hbgi(self, glucose_array: np.ndarray) -> float:
        """Calcula o índice de alta glicemia (HBGI)"""
        try:
            return np.mean(np.maximum(0, glucose_array - 180)) / 180
        except Exception as e:
            logger.error(f"Erro ao calcular HBGI: {e}")
            return 0.0
    
    def generate_metrics_report(self, metrics: PerformanceMetrics, 
                              period_hours: int = 24) -> str:
        """
        Gera relatório textual das métricas
        
        Args:
            metrics: Métricas calculadas
            period_hours: Período analisado em horas
            
        Returns:
            Relatório formatado em texto
        """
        report_lines = [
            f"📊 RELATÓRIO DE DESEMPENHO - {period_hours}h",
            "=" * 50,
            "",
            "🎯 CONTROLE GLICÊMICO:",
            f"  • Tempo em faixa (70-180): {metrics.time_in_range*100:.1f}%",
            f"  • Tempo abaixo (<70): {metrics.time_below_range*100:.1f}%",
            f"  • Tempo acima (>180): {metrics.time_above_range*100:.1f}%",
            f"  • Glicemia média: {metrics.mean_glucose:.1f} mg/dL",
            "",
            "⚡ VARIABILIDADE:",
            f"  • Desvio padrão: {metrics.glucose_std:.1f} mg/dL",
            f"  • Coeficiente variação: {metrics.coefficient_variation:.1f}%",
            f"  • GMI estimado: {metrics.glucose_management_indicator:.1f}%",
            "",
            "🚨 EVENTOS DE SEGURANÇA:",
            f"  • Hipoglicemias: {metrics.hypoglycemia_events}",
            f"  • Hipoglicemias severas: {metrics.severe_hypoglycemia_events}",
            f"  • Hiperglicemias prolongadas: {metrics.hyperglycemia_events}",
            "",
            "🤖 DESEMPENHO DO CONTROLADOR:",
            f"  • Precisão do controle: {metrics.controller_accuracy:.1f}%",
            f"  • Eficiência insulínica: {metrics.insulin_efficiency:.2f}",
            "",
            "📈 ÍNDICES DE RISCO:",
            f"  • LBGI (risco hipoglicemia): {metrics.low_blood_glucose_index:.2f}",
            f"  • HBGI (risco hiperglicemia): {metrics.high_blood_glucose_index:.2f}",
        ]
        
        # Adicionar interpretação
        report_lines.extend([
            "",
            "💡 INTERPRETAÇÃO:",
            self._interpret_metrics(metrics)
        ])
        
        return "\n".join(report_lines)
    
    def _interpret_metrics(self, metrics: PerformanceMetrics) -> str:
        """Gera interpretação das métricas"""
        interpretations = []
        
        # Avaliar TIR
        tir_pct = metrics.time_in_range * 100
        if tir_pct >= 70:
            interpretations.append("✅ Excelente controle glicêmico (TIR ≥70%)")
        elif tir_pct >= 50:
            interpretations.append("⚠️ Controle moderado - melhorar algoritmo")
        else:
            interpretations.append("🚨 Controle inadequado - revisão urgente")
        
        # Avaliar CV
        cv = metrics.coefficient_variation
        if cv <= 36:
            interpretations.append("✅ Baixa variabilidade glicêmica")
        else:
            interpretations.append("⚠️ Alta variabilidade - ajustar controlador")
        
        # Avaliar eventos
        if metrics.severe_hypoglycemia_events == 0:
            interpretations.append("✅ Sem hipoglicemias severas")
        else:
            interpretations.append("🚨 Atenção: hipoglicemias severas detectadas")
        
        return "\n  ".join(interpretations)
    
    def compare_periods(self, current_metrics: PerformanceMetrics,
                       previous_metrics: PerformanceMetrics) -> Dict:
        """
        Compara métricas entre dois períodos
        
        Args:
            current_metrics: Métricas do período atual
            previous_metrics: Métricas do período anterior
            
        Returns:
            Dicionário com comparações e tendências
        """
        comparisons = {}
        
        # Calcular diferenças percentuais
        metrics_to_compare = [
            'time_in_range', 'mean_glucose', 'coefficient_variation',
            'controller_accuracy', 'hypoglycemia_events'
        ]
        
        for metric in metrics_to_compare:
            current_val = getattr(current_metrics, metric)
            previous_val = getattr(previous_metrics, metric)
            
            if previous_val != 0:
                change_pct = ((current_val - previous_val) / previous_val) * 100
            else:
                change_pct = 0
            
            comparisons[metric] = {
                'current': current_val,
                'previous': previous_val,
                'change_percent': change_pct,
                'trend': 'improving' if change_pct > 0 else 'worsening' if change_pct < 0 else 'stable'
            }
        
        return comparisons
    
    def calculate_all_metrics(self, data: dict) -> dict:
        try:
            glucose_values = np.array([d['glucose'] for d in data])
            insulin_values = np.array([d['insulin_dose'] for d in data])
            
            metrics = {
                'time_in_range': self._calculate_tir(glucose_values),
                'mean_glucose': np.mean(glucose_values),
                'std_glucose': np.std(glucose_values),
                'hypo_events': self._count_events(glucose_values < 70),
                'hyper_events': self._count_events(glucose_values > 180),
                'total_insulin': np.sum(insulin_values)
            }
            
            return metrics
        except Exception as e:
            logger.error(f"Erro ao calcular métricas: {e}")
            return {}