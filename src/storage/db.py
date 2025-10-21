"""
Gerenciador de banco de dados para Sistema APS

Funcionalidades:
- Armazenamento de timesteps da simulação
- Histórico de decisões do controlador
- Logs de segurança
- Métricas calculadas
- Backup e exportação
"""

import logging
from pathlib import Path
import sqlite3
import aiosqlite
import json
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

# Configurar logger para este módulo
logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Gerenciador de banco de dados SQLite para o sistema APS
    
    Tabelas principais:
    - timesteps: Dados de cada step da simulação
    - decisions: Histórico de decisões do controlador
    - safety_events: Eventos de segurança
    - metrics: Métricas calculadas periodicamente
    """
    
    def __init__(self, config: dict):
        """
        Inicializa gerenciador de banco
        
        Args:
            config: Configuração do banco de dados
        """
        self.config = config
        self.db_path = config.get('path', 'data/aps_database.db')
        self.backup_interval_hours = config.get('backup_interval_hours', 24)
        
        self._ensure_db_exists()
        
        logger.info(f"Banco de dados: {self.db_path}")
    
    async def _ensure_db_exists(self):
        """Cria banco de dados e tabelas se não existirem"""
        Path(self.db_path).parent.mkdir(exist_ok=True)
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute('''
                CREATE TABLE IF NOT EXISTS timesteps (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    step INTEGER NOT NULL,
                    glucose REAL,
                    insulin_dose REAL,
                    dose_type TEXT,
                    prediction REAL,
                    confidence REAL,
                    pid_output REAL,
                    ml_adjustment REAL
                )
            ''')
            await db.commit()
        
        logger.info(f"Banco de dados: {self.db_path}")
    
    async def initialize(self):
        """Inicializa banco de dados e cria tabelas"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await self._create_tables(db)
                await db.commit()
            
            logger.info("Banco de dados inicializado com sucesso")
            
        except Exception as e:
            logger.error(f"Erro ao inicializar banco: {e}")
            raise
    
    async def _create_tables(self, db):
        """Cria todas as tabelas necessárias"""
        
        # Tabela de timesteps da simulação
        await db.execute("""
            CREATE TABLE IF NOT EXISTS timesteps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                step INTEGER NOT NULL,
                glucose REAL NOT NULL,
                glucose_true REAL,
                insulin_dose REAL DEFAULT 0,
                dose_type TEXT DEFAULT 'basal',
                prediction TEXT,  -- JSON com previsões
                confidence REAL DEFAULT 0,
                pid_output REAL DEFAULT 0,
                ml_adjustment REAL DEFAULT 0,
                iob REAL DEFAULT 0,
                controller_mode TEXT DEFAULT 'learning',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Tabela de decisões do controlador
        await db.execute("""
            CREATE TABLE IF NOT EXISTS controller_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                glucose REAL NOT NULL,
                target_glucose REAL DEFAULT 120,
                insulin_dose REAL NOT NULL,
                dose_type TEXT NOT NULL,
                pid_component REAL DEFAULT 0,
                ml_component REAL DEFAULT 0,
                reasoning TEXT,
                confidence REAL DEFAULT 0,
                safety_override BOOLEAN DEFAULT FALSE,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Tabela de eventos de segurança
        await db.execute("""
            CREATE TABLE IF NOT EXISTS safety_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                severity TEXT NOT NULL,  -- normal, warning, critical, emergency
                glucose REAL,
                message TEXT NOT NULL,
                action_taken TEXT,
                confidence REAL DEFAULT 1.0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Tabela de métricas calculadas
        await db.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                period_start TEXT NOT NULL,
                period_end TEXT NOT NULL,
                period_hours INTEGER NOT NULL,
                time_in_range REAL DEFAULT 0,
                mean_glucose REAL DEFAULT 0,
                glucose_std REAL DEFAULT 0,
                cv REAL DEFAULT 0,
                hypoglycemia_events INTEGER DEFAULT 0,
                hyperglycemia_events INTEGER DEFAULT 0,
                controller_accuracy REAL DEFAULT 0,
                additional_metrics TEXT,  -- JSON com métricas extras
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Tabela de refeições simuladas
        await db.execute("""
            CREATE TABLE IF NOT EXISTS meals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                carbs REAL NOT NULL,
                absorption_profile TEXT DEFAULT 'medium',
                gi_index REAL DEFAULT 50,
                fiber REAL DEFAULT 0,
                food_details TEXT,  -- JSON com detalhes dos alimentos
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Índices para performance
        await db.execute("CREATE INDEX IF NOT EXISTS idx_timesteps_timestamp ON timesteps(timestamp)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_decisions_timestamp ON controller_decisions(timestamp)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_safety_timestamp ON safety_events(timestamp)")
    
    async def save_timestep(self, timestep_data: Dict):
        """
        Salva dados de um timestep da simulação
        
        Args:
            timestep_data: Dados do timestep
        """
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT INTO timesteps (
                        timestamp, step, glucose, glucose_true, insulin_dose,
                        dose_type, prediction, confidence, pid_output,
                        ml_adjustment, iob, controller_mode
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    timestep_data['timestamp'].isoformat(),
                    timestep_data['step'],
                    timestep_data['glucose'],
                    timestep_data.get('glucose_true'),
                    timestep_data['insulin_dose'],
                    timestep_data['dose_type'],
                    json.dumps(timestep_data.get('prediction', [])),
                    timestep_data['confidence'],
                    timestep_data['pid_output'],
                    timestep_data['ml_adjustment'],
                    timestep_data.get('iob', 0),
                    timestep_data.get('controller_mode', 'learning')
                ))
                await db.commit()
                
        except Exception as e:
            logger.error(f"Erro ao salvar timestep: {e}")
    
    async def save_controller_decision(self, decision_data: Dict):
        """Salva decisão do controlador"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT INTO controller_decisions (
                        timestamp, glucose, target_glucose, insulin_dose,
                        dose_type, pid_component, ml_component, reasoning,
                        confidence, safety_override
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    decision_data['timestamp'].isoformat(),
                    decision_data['glucose'],
                    decision_data.get('target_glucose', 120),
                    decision_data['insulin_dose'],
                    decision_data['dose_type'],
                    decision_data['pid_component'],
                    decision_data['ml_component'],
                    decision_data['reasoning'],
                    decision_data['confidence'],
                    decision_data['safety_override']
                ))
                await db.commit()
                
        except Exception as e:
            logger.error(f"Erro ao salvar decisão: {e}")
    
    async def save_safety_event(self, event_data: Dict):
        """Salva evento de segurança"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT INTO safety_events (
                        timestamp, event_type, severity, glucose,
                        message, action_taken, confidence
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    event_data['timestamp'].isoformat(),
                    event_data['event_type'],
                    event_data['severity'],
                    event_data.get('glucose'),
                    event_data['message'],
                    event_data.get('action_taken', ''),
                    event_data.get('confidence', 1.0)
                ))
                await db.commit()
                
        except Exception as e:
            logger.error(f"Erro ao salvar evento de segurança: {e}")
    
    async def get_recent_data(self, hours: int = 24) -> List[Dict]:
        """
        Recupera dados recentes para análise
        
        Args:
            hours: Horas de histórico
            
        Returns:
            Lista de registros recentes
        """
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                
                cursor = await db.execute("""
                    SELECT * FROM timesteps 
                    WHERE timestamp >= ?
                    ORDER BY timestamp ASC
                """, (cutoff_time.isoformat(),))
                
                rows = await cursor.fetchall()
                
                # Converter para lista de dicionários
                data = []
                for row in rows:
                    record = dict(row)
                    record['timestamp'] = datetime.fromisoformat(record['timestamp'])
                    
                    # Decodificar JSON
                    if record['prediction']:
                        try:
                            record['prediction'] = json.loads(record['prediction'])
                        except json.JSONDecodeError:
                            record['prediction'] = []
                    
                    data.append(record)
                
                return data
                
        except Exception as e:
            logger.error(f"Erro ao recuperar dados: {e}")
            return []
    
    async def calculate_and_save_metrics(self, period_hours: int = 24):
        """
        Calcula e salva métricas para um período
        
        Args:
            period_hours: Período em horas
        """
        try:
            from utils.metrics import MetricsCalculator
            
            # Obter dados do período
            data = await self.get_recent_data(hours=period_hours)
            
            if not data:
                return
            
            # Calcular métricas
            calculator = MetricsCalculator()
            metrics = calculator.calculate_all_metrics(data)
            
            period_end = datetime.now()
            period_start = period_end - timedelta(hours=period_hours)
            
            # Salvar no banco
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT INTO metrics (
                        period_start, period_end, period_hours,
                        time_in_range, mean_glucose, glucose_std, cv,
                        hypoglycemia_events, hyperglycemia_events,
                        controller_accuracy, additional_metrics
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    period_start.isoformat(),
                    period_end.isoformat(),
                    period_hours,
                    metrics.time_in_range,
                    metrics.mean_glucose,
                    metrics.glucose_std,
                    metrics.coefficient_variation,
                    metrics.hypoglycemia_events,
                    metrics.hyperglycemia_events,
                    metrics.controller_accuracy,
                    json.dumps({
                        'gmi': metrics.glucose_management_indicator,
                        'lbgi': metrics.low_blood_glucose_index,
                        'hbgi': metrics.high_blood_glucose_index,
                        'insulin_efficiency': metrics.insulin_efficiency
                    })
                ))
                await db.commit()
                
            logger.info(f"Métricas calculadas e salvas para período de {period_hours}h")
                
        except Exception as e:
            logger.error(f"Erro ao calcular métricas: {e}")
    
    async def export_data(self, table_name: str, hours: int = None) -> str:
        """
        Exporta dados de uma tabela para CSV
        
        Args:
            table_name: Nome da tabela
            hours: Horas de histórico (None = todos os dados)
            
        Returns:
            Caminho do arquivo exportado
        """
        try:
            import pandas as pd
            
            async with aiosqlite.connect(self.db_path) as db:
                if hours:
                    cutoff_time = datetime.now() - timedelta(hours=hours)
                    query = f"SELECT * FROM {table_name} WHERE timestamp >= ? ORDER BY timestamp"
                    df = pd.read_sql_query(query, db, params=(cutoff_time.isoformat(),))
                else:
                    query = f"SELECT * FROM {table_name} ORDER BY timestamp"
                    df = pd.read_sql_query(query, db)
                
                # Gerar nome do arquivo
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"export_{table_name}_{timestamp}.csv"
                
                df.to_csv(filename, index=False)
                logger.info(f"Dados exportados para {filename}")
                
                return filename
                
        except Exception as e:
            logger.error(f"Erro ao exportar dados: {e}")
            return ""
    
    async def cleanup_old_data(self, days_to_keep: int = 30):
        """
        Remove dados antigos para manter o banco otimizado
        
        Args:
            days_to_keep: Dias de dados para manter
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            async with aiosqlite.connect(self.db_path) as db:
                # Contar registros antes
                cursor = await db.execute("SELECT COUNT(*) FROM timesteps WHERE timestamp < ?", 
                                        (cutoff_date.isoformat(),))
                count_before = (await cursor.fetchone())[0]
                
                if count_before > 0:
                    # Remover dados antigos
                    await db.execute("DELETE FROM timesteps WHERE timestamp < ?", 
                                   (cutoff_date.isoformat(),))
                    await db.execute("DELETE FROM controller_decisions WHERE timestamp < ?", 
                                   (cutoff_date.isoformat(),))
                    await db.execute("DELETE FROM safety_events WHERE timestamp < ?", 
                                   (cutoff_date.isoformat(),))
                    
                    await db.commit()
                    
                    logger.info(f"Removidos {count_before} registros antigos (>{days_to_keep} dias)")
                
        except Exception as e:
            logger.error(f"Erro na limpeza do banco: {e}")
            
# Exemplo de uso e teste das funcionalidades
if __name__ == "__main__":
    # Teste do catálogo de alimentos
    print("=== TESTE CATÁLOGO DE ALIMENTOS ===")
    catalog = get_food_catalog()
    print(f"Alimentos disponíveis: {len(catalog)}")
    
    # Exemplo de refeição
    meal_selections = [
        {'food': 'arroz_branco', 'amount_g': 150},
        {'food': 'feijao_preto', 'amount_g': 100},
        {'food': 'banana', 'amount_g': 120}
    ]
    
    meal_info = calculate_meal_carbs(meal_selections)
    print(f"\nRefeição: {meal_info['total_carbs']}g carb")
    print(f"Perfil: {meal_info['absorption_profile']}")
    print(f"IG médio: {meal_info['average_gi']}")
    
    # Sugestão de porções
    suggestions = suggest_meal_portions(45, ['cereais', 'frutas'])
    print(f"\nSugestões para 45g carb:")
    for s in suggestions:
        print(f"- {s['food_name']}: {s['amount_g']}g")
    
    print("\n=== TESTE SISTEMA DE MÉTRICAS ===")
    
    # Dados sintéticos para teste
    import numpy as np
    from datetime import datetime, timedelta
    
    # Simular 24h de dados (144 pontos, 10min cada)
    base_time = datetime.now() - timedelta(hours=24)
    test_data = []
    
    for i in range(144):
        timestamp = base_time + timedelta(minutes=i*10)
        
        # Simular glicemia com padrão circadiano e ruído
        hour = timestamp.hour
        circadian = 20 * np.sin(2 * np.pi * hour / 24) + 10 * np.sin(4 * np.pi * hour / 24)
        noise = np.random.normal(0, 15)
        glucose = 120 + circadian + noise
        
        # Ocasionalmente simular eventos
        if np.random.random() < 0.05:  # 5% chance
            glucose += np.random.normal(0, 40)  # Evento aleatório
        
        glucose = max(50, min(300, glucose))  # Limites fisiológicos
        
        test_data.append({
            'timestamp': timestamp,
            'glucose': glucose,
            'insulin_dose': max(0, np.random.normal(0.5, 0.2)),
            'step': i
        })
    
    # Calcular métricas
    calculator = MetricsCalculator()
    metrics = calculator.calculate_all_metrics(test_data)
    
    # Gerar relatório
    report = calculator.generate_metrics_report(metrics, 24)
    print(report)
    logger.warning("Dados insuficientes para calcular métricas")
    def calculate_metrics(self, data):
        """Calcula métricas com os dados fornecidos"""
        try:
            if not data:
                logger.warning("Dados insuficientes para calcular métricas")
                return self._empty_metrics()
            
            # Converter para arrays para processamento
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            glucose_values = df['glucose'].values
            timestamps = df['timestamp'].values
            
            # Métricas básicas de tempo em faixa
            tir_metrics = self._calculate_time_in_range(glucose_values)
            
            # Métricas estatísticas
            stats_metrics = self._calculate_statistical_metrics(glucose_values)
            
            # Métricas de eventos
            event_metrics = self._calculate_event_metrics(glucose_values, timestamps)
            
            # Métricas de qualidade
            quality_metrics = self._calculate_quality_metrics(glucose_values)
            
            # Métricas do controlador
            controller_metrics = self._calculate_controller_metrics(df)
            
            return PerformanceMetrics(
                **tir_metrics,
                **stats_metrics,
                **event_metrics,
                **quality_metrics,
                **controller_metrics
            )
            
        except Exception as e:
            logger.error(f"Erro ao calcular métricas: {e}")
            return self._empty_metrics()
    
    def _calculate_time_in_range(self, glucose_values: np.ndarray) -> Dict:
        """Calcula métricas de tempo em faixa"""
        total_readings = len(glucose_values)
        
        in_range = np.sum((glucose_values >= self.target_range[0]) & 
                         (glucose_values <= self.target_range[1]))
        below_range = np.sum(glucose_values < self.target_range[0])
        above_range = np.sum(glucose_values > self.target_range[1])
        
        return {
            'time_in_range': in_range / total_readings,
            'time_below_range': below_range / total_readings,
            'time_above_range': above_range / total_readings
        }
    
    def _calculate_statistical_metrics(self, glucose_values: np.ndarray) -> Dict:
        """Calcula métricas estatísticas"""
        mean_glucose = np.mean(glucose_values)
        std_glucose = np.std(glucose_values)
        cv = (std_glucose / mean_glucose) * 100 if mean_glucose > 0 else 0
        
        return {
            'mean_glucose': mean_glucose,
            'glucose_std': std_glucose,
            'coefficient_variation': cv
        }
    
    def _calculate_event_metrics(self, glucose_values: np.ndarray, 
                               timestamps: np.ndarray) -> Dict:
        """
        Calcula eventos de hipo/hiperglicemia
        
        Evento = período contínuo fora da faixa por tempo mínimo
        """
        # Detectar hipoglicemias (< 70 mg/dL por >= 15 min)
        hypo_mask = glucose_values < 70
        hypo_events = self._count_continuous_events(
            hypo_mask, timestamps, self.min_event_duration_min
        )
        
        # Hipoglicemias severas (< 54 mg/dL)
        severe_hypo_mask = glucose_values < self.severe_hypo_threshold
        severe_hypo_events = self._count_continuous_events(
            severe_hypo_mask, timestamps, self.min_event_duration_min
        )
        
        # Hiperglicemias (> 250 mg/dL por >= 2h)
        hyper_mask = glucose_values > self.severe_hyper_threshold
        hyper_events = self._count_continuous_events(
            hyper_mask, timestamps, self.hyper_event_duration_min
        )
        
        return {
            'hypoglycemia_events': hypo_events,
            'severe_hypoglycemia_events': severe_hypo_events,
            'hyperglycemia_events': hyper_events
        }
    
    def _count_continuous_events(self, mask: np.ndarray, timestamps: np.ndarray,
                               min_duration_min: int) -> int:
        """Conta eventos contínuos com duração mínima"""
        if len(mask) == 0:
            return 0
        
        events = 0
        in_event = False
        event_start_idx = None
        
        for i, is_event in enumerate(mask):
            if is_event and not in_event:
                # Início de evento
                in_event = True
                event_start_idx = i
            elif not is_event and in_event:
                # Fim de evento - verificar duração
                if event_start_idx is not None:
                    duration = (timestamps[i-1] - timestamps[event_start_idx])
                    duration_min = duration / np.timedelta64(1, 'm')
                    if duration_min >= min_duration_min:
                        events += 1
                in_event = False
                event_start_idx = None
        
        # Verificar evento que continua até o final
        if in_event and event_start_idx is not None:
            duration = (timestamps[-1] - timestamps[event_start_idx])
            duration_min = duration / np.timedelta64(1, 'm')
            if duration_min >= min_duration_min:
                events += 1
        
        return events
    
    def _calculate_quality_metrics(self, glucose_values: np.ndarray) -> Dict:
        """
        Calcula métricas de qualidade avançadas
        
        Baseado em:
        - Glucose Management Indicator (GMI)
        - Low/High Blood Glucose Index (LBGI/HBGI)
        """
        mean_glucose = np.mean(glucose_values)
        
        # GMI: correlação entre glucose média e HbA1c
        # GMI (%) = 3.31 + 0.02392 × mean_glucose_mg/dL
        gmi = 3.31 + 0.02392 * mean_glucose
        
        # LBGI e HBGI: índices de risco baseados em transformação logarítmica
        lbgi = self._calculate_lbgi(glucose_values)
        hbgi = self._calculate_hbgi(glucose_values)
        
        return {
            'glucose_management_indicator': gmi,
            'low_blood_glucose_index': lbgi,
            'high_blood_glucose_index': hbgi
        }
    
    def _calculate_lbgi(self, glucose_values: np.ndarray) -> float:
        """
        Calcula Low Blood Glucose Index
        
        Fórmula: LBGI = mean(max(0, 1.509 * (log(glucose)^1.084 - 5.381))^2)
        para glucose < 112.5 mg/dL
        """
        # Converter para mmol/L para cálculo padrão
        glucose_mmol = glucose_values / 18.0
        
        # Aplicar transformação apenas para valores baixos
        low_glucose_mask = glucose_mmol < 6.25  # ~112.5 mg/dL
        low_glucose = glucose_mmol[low_glucose_mask]
        
        if len(low_glucose) == 0:
            return 0.0
        
        # Fórmula LBGI
        log_glucose = np.log(low_glucose)
        transformed = 1.509 * (log_glucose ** 1.084 - 5.381)
        risk_values = np.maximum(0, transformed) ** 2
        
        lbgi = np.mean(risk_values)
        return lbgi
    
    def _calculate_hbgi(self, glucose_values: np.ndarray) -> float:
        """Calcula High Blood Glucose Index"""
        # Converter para mmol/L
        glucose_mmol = glucose_values / 18.0
        
        # Aplicar transformação apenas para valores altos
        high_glucose_mask = glucose_mmol > 6.25  # ~112.5 mg/dL
        high_glucose = glucose_mmol[high_glucose_mask]
        
        if len(high_glucose) == 0:
            return 0.0
        
        # Fórmula HBGI
        log_glucose = np.log(high_glucose)
        transformed = 1.509 * (log_glucose ** 1.084 - 5.381)
        risk_values = np.maximum(0, transformed) ** 2
        
        hbgi = np.mean(risk_values)
        return hbgi
    
    def _calculate_controller_metrics(self, df: pd.DataFrame) -> Dict:
        """Calcula métricas específicas do controlador"""
        try:
            glucose_values = df['glucose'].values
            target_glucose = 120.0  # mg/dL alvo
            
            # Accuracy: MARD entre glucose atual e alvo
            mard = np.mean(np.abs(glucose_values - target_glucose) / target_glucose) * 100
            controller_accuracy = max(0, 100 - mard)  # Converter para % de precisão
            
            # Eficiência da insulina: redução de glucose por unidade
            insulin_efficiency = 0.0
            if 'insulin_dose' in df.columns:
                insulin_doses = df['insulin_dose'].values
                total_insulin = np.sum(insulin_doses)
                
                if total_insulin > 0:
                    # Correlação entre doses e redução da hiperglicemia
                    hyper_reduction = np.sum(np.maximum(0, glucose_values - 180))
                    insulin_efficiency = hyper_reduction / total_insulin if total_insulin > 0 else 0
            
            return {
                'controller_accuracy': controller_accuracy,
                'insulin_efficiency': insulin_efficiency
            }
            
        except Exception as e:
            logger.error(f"Erro ao calcular métricas do controlador: {e}")
            return {
                'controller_accuracy': 0.0,
                'insulin_efficiency': 0.0
            }