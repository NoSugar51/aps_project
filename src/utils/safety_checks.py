"""
Sistema de Verificações de Segurança para APS

Implementa múltiplas camadas de segurança críticas:
- Detecção de hipoglicemia iminente
- Limites de dose de insulina
- Verificação de integridade do sensor
- Detecção de falhas de comunicação
- Alertas e paradas de emergência

⚠️ CRÍTICO: Este módulo implementa salvaguardas essenciais para segurança.
Qualquer modificação deve ser cuidadosamente testada e validada.
"""

import logging
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)

# Configurar logger para este módulo
# logger = logging.getLogger(__name__)

@dataclass
class SafetyCheck:
    """Resultado de verificação de segurança"""
    is_safe: bool
    message: str = ""
    emergency_stop: bool = False

class SafetyLevel(Enum):
    """Níveis de segurança do sistema"""
    NORMAL = "normal"
    WARNING = "warning" 
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class AlertType(Enum):
    """Tipos de alertas de segurança"""
    HYPOGLYCEMIA = "hypoglycemia"
    HYPERGLYCEMIA = "hyperglycemia"
    SENSOR_ERROR = "sensor_error"
    INSULIN_OVERDOSE = "insulin_overdose"
    COMMUNICATION_FAILURE = "communication_failure"
    SYSTEM_ERROR = "system_error"

@dataclass
class SafetyCheck:
    """Resultado de verificação de segurança"""
    is_safe: bool
    safety_level: SafetyLevel
    alert_type: Optional[AlertType]
    message: str
    emergency_stop: bool = False
    recommended_action: str = ""
    confidence: float = 1.0  # Confiança na decisão (0-1)

class SafetyMonitor:
    """
    Monitor de segurança para sistema APS
    
    Implementa verificações contínuas de segurança baseadas em:
    - Guidelines clínicos para sistemas de pâncreas artificial
    - Padrões de segurança para dispositivos médicos críticos
    - Literatura sobre prevenção de hipoglicemia severa
    """
    
    def __init__(self, safety_config: dict):
        """
        Inicializa monitor de segurança
        
        Args:
            safety_config: Configuração dos limites de segurança
        """
        self.config = safety_config
        
        # Limites críticos de glicemia
        self.min_glucose_alarm = safety_config.get('min_glucose_alarm', 70)  # mg/dL
        self.max_glucose_alarm = safety_config.get('max_glucose_alarm', 250)  # mg/dL
        self.critical_low_glucose = 55  # mg/dL - hipoglicemia severa
        self.critical_high_glucose = 400  # mg/dL - hiperglicemia severa
        
        # Limites de insulina
        self.max_bolus_per_30min = safety_config.get('max_bolus_per_30min', 4.0)  # U
        self.max_daily_total = safety_config.get('max_daily_total', 100.0)  # U
        self.max_iob_warning = 6.0  # U - IOB alto
        self.max_iob_critical = 10.0  # U - IOB perigoso
        
        # Limites do sensor
        self.sensor_spike_threshold = safety_config.get('sensor_spike_threshold', 40)  # mg/dL
        self.max_sensor_age_hours = 240  # 10 dias máximo
        self.min_readings_per_hour = 3  # Mínimo de leituras
        
        # Timeouts de comunicação
        self.connection_timeout = safety_config.get('connection_timeout_minutes', 30)
        
        # Histórico de alertas
        self.alert_history: List[Dict] = []
        self.last_glucose_reading = None
        self.last_communication_time = datetime.now()
        
        # Estado do sistema
        self.system_state = SafetyLevel.NORMAL
        self.emergency_mode = False
        
        logger.info("Monitor de segurança inicializado")
        logger.info(f"Limites: Glicemia {self.min_glucose_alarm}-{self.max_glucose_alarm} mg/dL")
        logger.info(f"Limites: Bolus máx {self.max_bolus_per_30min}U/30min")
    
    def check_safety(self, glucose: float, insulin_history: List[dict],
                    sensor_data: dict) -> SafetyCheck:
        """
        Verificação principal de segurança
        
        Args:
            glucose: Glicemia atual em mg/dL
            insulin_history: Histórico de doses de insulina
            sensor_data: Dados do sensor
            
        Returns:
            SafetyCheck com resultado da avaliação
        """
        try:
            # Lista de verificações a executar
            checks = [
                self._check_glucose_levels(glucose),
                self._check_glucose_trend(glucose, sensor_data),
                self._check_insulin_safety(insulin_history),
                self._check_sensor_integrity(glucose, sensor_data),
                self._check_communication_status(),
                self._check_hypoglycemia_prediction(glucose, sensor_data)
            ]
            
            # Encontrar o alerta mais crítico
            most_critical = self._get_most_critical_check(checks)
            
            # Atualizar estado do sistema
            self._update_system_state(most_critical)
            
            # Registrar no histórico
            self._log_safety_check(most_critical, glucose)
            
            return most_critical
            
        except Exception as e:
            logger.error(f"Erro na verificação de segurança: {e}")
            return SafetyCheck(
                is_safe=False,
                safety_level=SafetyLevel.EMERGENCY,
                alert_type=AlertType.SYSTEM_ERROR,
                message="🚨 Erro crítico no sistema de segurança",
                emergency_stop=True,
                recommended_action="Intervenção manual imediata necessária"
            )
    
    def _check_glucose_levels(self, glucose: float) -> SafetyCheck:
        """
        Verifica níveis críticos de glicemia
        
        Args:
            glucose: Glicemia atual
            
        Returns:
            SafetyCheck para níveis glicêmicos
        """
        if glucose <= self.critical_low_glucose:
            return SafetyCheck(
                is_safe=False,
                safety_level=SafetyLevel.EMERGENCY,
                alert_type=AlertType.HYPOGLYCEMIA,
                message=f"🚨 HIPOGLICEMIA SEVERA: {glucose:.1f} mg/dL",
                emergency_stop=True,
                recommended_action="Administrar glicose imediatamente",
                confidence=0.95
            )
        elif glucose <= self.min_glucose_alarm:
            return SafetyCheck(
                is_safe=False,
                safety_level=SafetyLevel.CRITICAL,
                alert_type=AlertType.HYPOGLYCEMIA,
                message=f"⚠️ Hipoglicemia: {glucose:.1f} mg/dL",
                emergency_stop=False,
                recommended_action="Reduzir/suspender insulina, monitorar de perto",
                confidence=0.9
            )
        elif glucose >= self.critical_high_glucose:
            return SafetyCheck(
                is_safe=False,
                safety_level=SafetyLevel.EMERGENCY,
                alert_type=AlertType.HYPERGLYCEMIA,
                message=f"🚨 HIPERGLICEMIA SEVERA: {glucose:.1f} mg/dL",
                emergency_stop=False,
                recommended_action="Verificar cetonas, considerar correção",
                confidence=0.85
            )
        elif glucose >= self.max_glucose_alarm:
            return SafetyCheck(
                is_safe=False,
                safety_level=SafetyLevel.WARNING,
                alert_type=AlertType.HYPERGLYCEMIA,
                message=f"⚠️ Hiperglicemia: {glucose:.1f} mg/dL",
                emergency_stop=False,
                recommended_action="Monitorar tendência, considerar correção",
                confidence=0.8
            )
        else:
            return SafetyCheck(
                is_safe=True,
                safety_level=SafetyLevel.NORMAL,
                alert_type=None,
                message=f"✅ Glicemia normal: {glucose:.1f} mg/dL",
                recommended_action="Continuar monitoramento normal"
            )
    
    def _check_glucose_trend(self, glucose: float, sensor_data: dict) -> SafetyCheck:
        """
        Verifica tendência glicêmica perigosa
        
        Args:
            glucose: Glicemia atual
            sensor_data: Dados do sensor com tendência
            
        Returns:
            SafetyCheck para tendência glicêmica
        """
        trend = sensor_data.get('trend', 0)  # mg/dL por leitura
        
        # Queda rápida (risco de hipoglicemia)
        if trend < -5 and glucose < 100:
            severity = SafetyLevel.CRITICAL if glucose < 80 else SafetyLevel.WARNING
            return SafetyCheck(
                is_safe=False,
                safety_level=severity,
                alert_type=AlertType.HYPOGLYCEMIA,
                message=f"⚠️ Queda rápida: {trend:.1f} mg/dL/leitura",
                emergency_stop=(severity == SafetyLevel.CRITICAL),
                recommended_action="Suspender insulina, preparar glicose",
                confidence=0.8
            )
        
        # Subida muito rápida (possível erro de sensor)
        elif trend > 8:
            return SafetyCheck(
                is_safe=False,
                safety_level=SafetyLevel.WARNING,
                alert_type=AlertType.SENSOR_ERROR,
                message=f"⚠️ Subida muito rápida: {trend:.1f} mg/dL/leitura",
                emergency_stop=False,
                recommended_action="Verificar sensor, confirmar com medição manual",
                confidence=0.7
            )
        
        return SafetyCheck(
            is_safe=True,
            safety_level=SafetyLevel.NORMAL,
            alert_type=None,
            message="✅ Tendência glicêmica estável"
        )
    
    def _check_insulin_safety(self, insulin_history: List[dict]) -> SafetyCheck:
        """
        Verifica limites de segurança da insulina
        
        Args:
            insulin_history: Histórico de doses
            
        Returns:
            SafetyCheck para doses de insulina
        """
        if not insulin_history:
            return SafetyCheck(is_safe=True, safety_level=SafetyLevel.NORMAL, 
                             alert_type=None, message="✅ Nenhuma dose recente")
        
        now = datetime.now()
        
        # Verificar doses nos últimos 30 minutos
        recent_30min = [
            dose for dose in insulin_history
            if (now - dose['timestamp']).total_seconds() <= 1800
        ]
        
        total_30min = sum(dose['dose'] for dose in recent_30min)
        
        if total_30min > self.max_bolus_per_30min:
            return SafetyCheck(
                is_safe=False,
                safety_level=SafetyLevel.CRITICAL,
                alert_type=AlertType.INSULIN_OVERDOSE,
                message=f"🚨 Excesso de insulina: {total_30min:.1f}U em 30min",
                emergency_stop=True,
                recommended_action="PARAR todas as doses, monitorar hipoglicemia",
                confidence=0.95
            )
        
        # Verificar doses diárias
        daily_cutoff = now - timedelta(hours=24)
        daily_doses = [
            dose for dose in insulin_history
            if dose['timestamp'] >= daily_cutoff
        ]
        
        total_daily = sum(dose['dose'] for dose in daily_doses)
        
        if total_daily > self.max_daily_total:
            return SafetyCheck(
                is_safe=False,
                safety_level=SafetyLevel.WARNING,
                alert_type=AlertType.INSULIN_OVERDOSE,
                message=f"⚠️ Alto total diário: {total_daily:.1f}U/24h",
                emergency_stop=False,
                recommended_action="Revisar configurações, reduzir doses futuras",
                confidence=0.8
            )
        
        # Verificar IOB (Insulin on Board)
        current_iob = self._calculate_current_iob(insulin_history)
        
        if current_iob > self.max_iob_critical:
            return SafetyCheck(
                is_safe=False,
                safety_level=SafetyLevel.CRITICAL,
                alert_type=AlertType.INSULIN_OVERDOSE,
                message=f"🚨 IOB crítico: {current_iob:.1f}U ativo",
                emergency_stop=True,
                recommended_action="SUSPENDER insulina, monitorar hipoglicemia de perto",
                confidence=0.9
            )
        elif current_iob > self.max_iob_warning:
            return SafetyCheck(
                is_safe=False,
                safety_level=SafetyLevel.WARNING,
                alert_type=AlertType.INSULIN_OVERDOSE,
                message=f"⚠️ IOB elevado: {current_iob:.1f}U ativo",
                emergency_stop=False,
                recommended_action="Reduzir próximas doses, monitorar glicemia",
                confidence=0.85
            )
        
        return SafetyCheck(
            is_safe=True,
            safety_level=SafetyLevel.NORMAL,
            alert_type=None,
            message=f"✅ Insulina segura: IOB {current_iob:.1f}U"
        )
    
    def _calculate_current_iob(self, insulin_history: List[dict]) -> float:
        """
        Calcula insulina ativa atual (IOB)
        
        Args:
            insulin_history: Histórico de doses
            
        Returns:
            IOB em unidades
        """
        iob = 0.0
        now = datetime.now()
        tau = 60  # Constante de tempo em minutos
        
        for dose in insulin_history:
            elapsed_minutes = (now - dose['timestamp']).total_seconds() / 60.0
            
            if 0 < elapsed_minutes < 300:  # Máximo 5 horas
                # Modelo exponencial de absorção
                remaining_fraction = np.exp(-elapsed_minutes / tau)
                iob += dose['dose'] * remaining_fraction
        
        return iob
    
    def _check_sensor_integrity(self, glucose: float, sensor_data: dict) -> SafetyCheck:
        """
        Verifica integridade e qualidade do sensor
        
        Args:
            glucose: Leitura atual
            sensor_data: Dados do sensor
            
        Returns:
            SafetyCheck para sensor
        """
        # Verificar spike repentino
        if self.last_glucose_reading is not None:
            glucose_change = abs(glucose - self.last_glucose_reading)
            if glucose_change > self.sensor_spike_threshold:
                return SafetyCheck(
                    is_safe=False,
                    safety_level=SafetyLevel.WARNING,
                    alert_type=AlertType.SENSOR_ERROR,
                    message=f"⚠️ Spike no sensor: {glucose_change:.1f} mg/dL",
                    emergency_stop=False,
                    recommended_action="Verificar sensor, confirmar com teste manual",
                    confidence=0.7
                )
        
        # Verificar qualidade do sensor
        sensor_quality = sensor_data.get('quality', 'good')
        if sensor_quality == 'poor':
            return SafetyCheck(
                is_safe=False,
                safety_level=SafetyLevel.WARNING,
                alert_type=AlertType.SENSOR_ERROR,
                message="⚠️ Qualidade do sensor baixa",
                emergency_stop=False,
                recommended_action="Calibrar ou substituir sensor",
                confidence=0.6
            )
        
        # Verificar valores fora da faixa fisiológica
        if glucose < 20 or glucose > 600:
            return SafetyCheck(
                is_safe=False,
                safety_level=SafetyLevel.CRITICAL,
                alert_type=AlertType.SENSOR_ERROR,
                message=f"🚨 Leitura implausível: {glucose:.1f} mg/dL",
                emergency_stop=True,
                recommended_action="Verificar sensor imediatamente, medir manualmente",
                confidence=0.95
            )
        
        self.last_glucose_reading = glucose
        
        return SafetyCheck(
            is_safe=True,
            safety_level=SafetyLevel.NORMAL,
            alert_type=None,
            message="✅ Sensor funcionando normalmente"
        )
    
    def _check_communication_status(self) -> SafetyCheck:
        """
        Verifica status de comunicação com dispositivos
        
        Returns:
            SafetyCheck para comunicação
        """
        time_since_last_comm = (datetime.now() - self.last_communication_time).total_seconds() / 60
        
        if time_since_last_comm > self.connection_timeout:
            return SafetyCheck(
                is_safe=False,
                safety_level=SafetyLevel.CRITICAL,
                alert_type=AlertType.COMMUNICATION_FAILURE,
                message=f"🚨 Perda de comunicação: {time_since_last_comm:.0f} min",
                emergency_stop=True,
                recommended_action="Verificar conexões, modo manual se necessário",
                confidence=0.9
            )
        elif time_since_last_comm > self.connection_timeout / 2:
            return SafetyCheck(
                is_safe=False,
                safety_level=SafetyLevel.WARNING,
                alert_type=AlertType.COMMUNICATION_FAILURE,
                message=f"⚠️ Comunicação instável: {time_since_last_comm:.0f} min",
                emergency_stop=False,
                recommended_action="Verificar qualidade da conexão",
                confidence=0.8
            )
        
        return SafetyCheck(
            is_safe=True,
            safety_level=SafetyLevel.NORMAL,
            alert_type=None,
            message="✅ Comunicação estável"
        )
    
    def _check_hypoglycemia_prediction(self, glucose: float, sensor_data: dict) -> SafetyCheck:
        """
        Prediz risco de hipoglicemia iminente
        
        Args:
            glucose: Glicemia atual
            sensor_data: Dados do sensor
            
        Returns:
            SafetyCheck para predição de hipoglicemia
        """
        trend = sensor_data.get('trend', 0)
        
        # Modelo simples: extrapolar tendência linear
        predicted_glucose_10min = glucose + (trend * 1)  # 1 leitura ≈ 10 min
        predicted_glucose_20min = glucose + (trend * 2)
        
        # Verificar risco de hipoglicemia nos próximos 20 min
        if predicted_glucose_20min < self.critical_low_glucose:
            return SafetyCheck(
                is_safe=False,
                safety_level=SafetyLevel.CRITICAL,
                alert_type=AlertType.HYPOGLYCEMIA,
                message=f"🚨 Hipoglicemia prevista: {predicted_glucose_20min:.1f} mg/dL em 20min",
                emergency_stop=True,
                recommended_action="PARAR insulina, preparar glicose de emergência",
                confidence=0.75
            )
        elif predicted_glucose_10min < self.min_glucose_alarm:
            return SafetyCheck(
                is_safe=False,
                safety_level=SafetyLevel.WARNING,
                alert_type=AlertType.HYPOGLYCEMIA,
                message=f"⚠️ Risco de hipoglicemia: {predicted_glucose_10min:.1f} mg/dL em 10min",
                emergency_stop=False,
                recommended_action="Suspender insulina, monitorar de perto",
                confidence=0.65
            )
        
        return SafetyCheck(
            is_safe=True,
            safety_level=SafetyLevel.NORMAL,
            alert_type=None,
            message="✅ Sem risco iminente de hipoglicemia"
        )
    
    def _get_most_critical_check(self, checks: List[SafetyCheck]) -> SafetyCheck:
        """
        Determina a verificação mais crítica
        
        Args:
            checks: Lista de verificações
            
        Returns:
            Verificação mais crítica
        """
        # Ordenar por nível de criticidade
        level_priority = {
            SafetyLevel.EMERGENCY: 4,
            SafetyLevel.CRITICAL: 3,
            SafetyLevel.WARNING: 2,
            SafetyLevel.NORMAL: 1
        }
        
        most_critical = max(checks, key=lambda x: level_priority[x.safety_level])
        
        return most_critical
    
    def _update_system_state(self, check: SafetyCheck):
        """
        Atualiza estado geral do sistema
        
        Args:
            check: Verificação mais crítica
        """
        self.system_state = check.safety_level
        
        if check.emergency_stop:
            self.emergency_mode = True
            logger.critical(f"MODO DE EMERGÊNCIA ATIVADO: {check.message}")
        elif check.safety_level == SafetyLevel.NORMAL and self.emergency_mode:
            self.emergency_mode = False
            logger.info("Modo de emergência desativado - sistema normalizado")
    
    def _log_safety_check(self, check: SafetyCheck, glucose: float):
        """
        Registra verificação no histórico
        
        Args:
            check: Verificação realizada
            glucose: Glicemia atual
        """
        log_entry = {
            'timestamp': datetime.now(),
            'glucose': glucose,
            'is_safe': check.is_safe,
            'safety_level': check.safety_level.value,
            'alert_type': check.alert_type.value if check.alert_type else None,
            'message': check.message,
            'emergency_stop': check.emergency_stop,
            'confidence': check.confidence
        }
        
        self.alert_history.append(log_entry)
        
        # Manter histórico limitado
        if len(self.alert_history) > 1000:
            self.alert_history.pop(0)
        
        # Log baseado no nível
        if check.safety_level == SafetyLevel.EMERGENCY:
            logger.critical(f"SEGURANÇA: {check.message}")
        elif check.safety_level == SafetyLevel.CRITICAL:
            logger.error(f"SEGURANÇA: {check.message}")
        elif check.safety_level == SafetyLevel.WARNING:
            logger.warning(f"SEGURANÇA: {check.message}")
        else:
            logger.debug(f"SEGURANÇA: {check.message}")
    
    def update_communication_status(self):
        """Atualiza timestamp da última comunicação"""
        self.last_communication_time = datetime.now()
    
    def force_emergency_stop(self, reason: str):
        """
        Força parada de emergência do sistema
        
        Args:
            reason: Motivo da parada
        """
        self.emergency_mode = True
        self.system_state = SafetyLevel.EMERGENCY
        
        log_entry = {
            'timestamp': datetime.now(),
            'glucose': None,
            'is_safe': False,
            'safety_level': SafetyLevel.EMERGENCY.value,
            'alert_type': AlertType.SYSTEM_ERROR.value,
            'message': f"🚨 PARADA FORÇADA: {reason}",
            'emergency_stop': True,
            'confidence': 1.0
        }
        
        self.alert_history.append(log_entry)
        logger.critical(f"PARADA DE EMERGÊNCIA FORÇADA: {reason}")
    
    def reset_emergency_mode(self):
        """
        Reseta modo de emergência (usar com cuidado!)
        
        Só deve ser usado após verificação manual completa
        """
        self.emergency_mode = False
        self.system_state = SafetyLevel.NORMAL
        logger.warning("Modo de emergência resetado manualmente")
    
    def get_safety_metrics(self, hours: int = 24) -> dict:
        """
        Calcula métricas de segurança
        
        Args:
            hours: Período para análise
            
        Returns:
            Métricas de segurança
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_alerts = [
            alert for alert in self.alert_history
            if alert['timestamp'] >= cutoff_time
        ]
        
        if not recent_alerts:
            return {}
        
        total_checks = len(recent_alerts)
        
        # Contadores por tipo
        level_counts = {}
        alert_type_counts = {}
        
        for alert in recent_alerts:
            level = alert['safety_level']
            level_counts[level] = level_counts.get(level, 0) + 1
            
            if alert['alert_type']:
                alert_type = alert['alert_type']
                alert_type_counts[alert_type] = alert_type_counts.get(alert_type, 0) + 1
        
        # Calcular taxas
        emergency_rate = level_counts.get('emergency', 0) / total_checks
        critical_rate = level_counts.get('critical', 0) / total_checks
        warning_rate = level_counts.get('warning', 0) / total_checks
        normal_rate = level_counts.get('normal', 0) / total_checks
        
        return {
            'total_safety_checks': total_checks,
            'emergency_rate': emergency_rate,
            'critical_rate': critical_rate,
            'warning_rate': warning_rate,
            'normal_rate': normal_rate,
            'current_system_state': self.system_state.value,
            'emergency_mode_active': self.emergency_mode,
            'alert_type_distribution': alert_type_counts,
            'most_common_alert': max(alert_type_counts, key=alert_type_counts.get) if alert_type_counts else None,
            'average_confidence': np.mean([a['confidence'] for a in recent_alerts])
        }
    
    def export_safety_log(self, filename: str = None) -> str:
        """
        Exporta log de segurança
        
        Args:
            filename: Nome do arquivo
            
        Returns:
            Caminho do arquivo criado
        """
        import pandas as pd
        
        if filename is None:
            filename = f"safety_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        if not self.alert_history:
            logger.warning("Nenhum dado de segurança para exportar")
            return filename
        
        # Converter para DataFrame
        df = pd.DataFrame(self.alert_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        df.to_csv(filename, index=False)
        
        logger.info(f"Log de segurança exportado para {filename}")
        return filename

# Funções utilitárias para segurança

def validate_glucose_reading(glucose: float) -> bool:
    """
    Valida se leitura de glicose é fisiologicamente plausível
    
    Args:
        glucose: Valor de glicose em mg/dL
        
    Returns:
        True se plausível, False caso contrário
    """
    return 20 <= glucose <= 600

def estimate_hypoglycemia_risk(glucose: float, trend: float, iob: float) -> float:
    """
    Estima risco de hipoglicemia nos próximos 30 minutos
    
    Args:
        glucose: Glicemia atual
        trend: Tendência (mg/dL por período)
        iob: Insulina ativa
        
    Returns:
        Risco entre 0 e 1 (0=sem risco, 1=risco máximo)
    """
    # Fatores de risco
    glucose_factor = max(0, (90 - glucose) / 90)  # Risco aumenta abaixo de 90
    trend_factor = max(0, -trend / 20)  # Risco aumenta com tendência negativa
    iob_factor = min(1, iob / 5)  # Risco aumenta com IOB alto
    
    # Combinação ponderada
    risk = 0.5 * glucose_factor + 0.3 * trend_factor + 0.2 * iob_factor
    
    return min(1.0, risk)

def calculate_insulin_duration_remaining(dose_history: List[dict]) -> dict:
    """
    Calcula tempo restante de ação das doses de insulina
    
    Args:
        dose_history: Histórico de doses
        
    Returns:
        Dicionário com informações de duração
    """
    now = datetime.now()
    active_doses = []
    
    for dose in dose_history:
        elapsed_hours = (now - dose['timestamp']).total_seconds() / 3600
        
        # Insulina rápida: duração ~4-6 horas
        if dose.get('type') == 'bolus':
            duration = 5.0  # horas
        else:  # basal
            duration = 1.0  # horas
        
        remaining = duration - elapsed_hours
        
        if remaining > 0:
            active_doses.append({
                'dose': dose['dose'],
                'type': dose.get('type', 'unknown'),
                'remaining_hours': remaining,
                'effectiveness': max(0, remaining / duration)
            })
    
    total_remaining_effect = sum(d['dose'] * d['effectiveness'] for d in active_doses)
    
    return {
        'active_doses': len(active_doses),
        'total_remaining_units': sum(d['dose'] for d in active_doses),
        'total_remaining_effect': total_remaining_effect,
        'longest_duration_hours': max([d['remaining_hours'] for d in active_doses], default=0)
    }