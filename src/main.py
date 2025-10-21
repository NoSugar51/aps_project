#!/usr/bin/env python3
"""
Sistema de Pâncreas Artificial (APS) - Ponto de Entrada Principal

⚠️ AVISO LEGAL: Sistema para pesquisa e simulação apenas.
NÃO USAR EM PACIENTES REAIS sem validação clínica e aprovação regulatória.

Autor: Sistema APS
Data: 2025
"""

# Configuração inicial do logger
import logging
import logging.config
import sys
from pathlib import Path

Path('logs').mkdir(exist_ok=True)

# Configuração do logger mais robusta
logging_config = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'default': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        }
    },
    'handlers': {
        'file': {
            'class': 'logging.FileHandler',
            'filename': 'logs/aps_system.log',
            'formatter': 'default'
        },
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'default'
        }
    },
    'root': {
        'handlers': ['file', 'console'],
        'level': 'INFO'
    }
}

# Aplicar configuração
logging.config.dictConfig(logging_config)
logger = logging.getLogger(__name__)

# Importações
import argparse
import asyncio
import yaml
import uvicorn
from fastapi import FastAPI

# Criar diretórios necessários
Path('logs').mkdir(exist_ok=True)
Path('data').mkdir(exist_ok=True)
Path('models').mkdir(exist_ok=True)
Path('exports').mkdir(exist_ok=True)

try:
    # Importações locais com tratamento de erro
    from controller.hybrid_controller import HybridController
    from sim.simulator import APSSimulator
    from ml.trainer import MLTrainer
    from storage.db import DatabaseManager
    from api.rest import create_app
    from utils.safety_checks import SafetyMonitor
    from utils.metrics import MetricsCalculator
except Exception as e:
    logger.error(f"Erro ao importar módulos: {e}")
    sys.exit(1)

class APSSystem:
    """Sistema principal de Pâncreas Artificial"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Inicializa o sistema APS
        
        Args:
            config_path: Caminho para arquivo de configuração
        """
        self.config = self._load_config(config_path)
        self.running = False
        
        # Inicializar componentes principais
        self._init_components()
        
        logger.info("Sistema APS inicializado com sucesso")
        
    def _load_config(self, config_path: str) -> dict:
        """Carrega configuração do arquivo YAML"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuração carregada de {config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"Arquivo de configuração não encontrado: {config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Erro ao ler configuração YAML: {e}")
            raise
            
    def _init_components(self):
        """Inicializa todos os componentes do sistema"""
        
        # Database
        self.db = DatabaseManager(self.config['database'])
        
        # Safety monitor
        self.safety_monitor = SafetyMonitor(self.config['safety'])
        
        # Metrics calculator
        self.metrics = MetricsCalculator()
        
        # ML Trainer
        self.ml_trainer = MLTrainer(self.config['ml'])
        
        # Controlador híbrido
        self.controller = HybridController(
            config=self.config['controller'],
            ml_model=self.ml_trainer.get_model(),
            safety_monitor=self.safety_monitor
        )
        
        # Simulador
        self.simulator = APSSimulator(
            config=self.config['simulation'],
            patient_config=self.config['patient']
        )
        
        logger.info("Todos os componentes inicializados")
    
    async def run_simulation(self, duration_hours: int = 24, acceleration: int = 1):
        """
        Executa simulação do sistema APS
        
        Args:
            duration_hours: Duração da simulação em horas
            acceleration: Fator de aceleração temporal (1-100x)
        """
        logger.info(f"Iniciando simulação: {duration_hours}h com aceleração {acceleration}x")
        
        self.running = True
        self.simulator.set_time_acceleration(acceleration)
        
        try:
            simulation_steps = int((duration_hours * 60) / self.config['simulation']['time_step_minutes'])
            
            for step in range(simulation_steps):
                if not self.running:
                    break
                    
                # Obter leitura atual do sensor
                current_glucose = self.simulator.get_current_glucose()
                
                # Verificações de segurança
                safety_check = self.safety_monitor.check_safety(
                    glucose=current_glucose,
                    insulin_history=self.simulator.get_insulin_history(),
                    sensor_data=self.simulator.get_sensor_data()
                )
                
                if not safety_check.is_safe:
                    logger.warning(f"Alerta de segurança: {safety_check.message}")
                    if safety_check.emergency_stop:
                        logger.critical("PARADA DE EMERGÊNCIA ATIVADA")
                        break
                
                # Calcular dose de insulina
                control_output = self.controller.calculate_dose(
                    current_glucose=current_glucose,
                    glucose_history=self.simulator.get_glucose_history(),
                    insulin_history=self.simulator.get_insulin_history(),
                    meal_history=self.simulator.get_meal_history()
                )
                
                # Aplicar dose se segura
                if safety_check.is_safe and control_output.dose > 0:
                    self.simulator.apply_insulin_dose(
                        dose=control_output.dose,
                        dose_type=control_output.dose_type
                    )
                
                # Avançar simulação
                self.simulator.step()
                
                # Salvar dados
                await self._save_timestep_data(step, current_glucose, control_output)
                
                # Treinamento ML incremental (a cada hora)
                if step % 6 == 0:  # 6 steps = 1 hora (10min/step)
                    await self._incremental_training()
                
                # Log progresso
                if step % 30 == 0:  # A cada 5 horas
                    progress = (step / simulation_steps) * 100
                    logger.info(f"Progresso simulação: {progress:.1f}%")
                    
            logger.info("Simulação concluída com sucesso")
            
        except Exception as e:
            logger.error(f"Erro durante simulação: {e}")
            raise
        finally:
            self.running = False
    
    async def run_real_mode(self):
        """
        Executa sistema em modo real com hardware
        
        ⚠️ ATENÇÃO: Modo real requer validação clínica!
        """
        logger.warning("MODO REAL - REQUER VALIDAÇÃO CLÍNICA")
        
        # Importar adaptadores de hardware
        try:
            from hardware.dana_adapter import DanaAdapter
            from hardware.libre2_adapter import Libre2Adapter
            
            # Inicializar hardware
            pump = DanaAdapter(self.config['hardware']['dana_pump'])
            sensor = Libre2Adapter(self.config['hardware']['libre2_sensor'])
            
            await pump.connect()
            await sensor.connect()
            
            logger.info("Hardware conectado - iniciando modo real")
            
            self.running = True
            
            while self.running:
                # Loop principal do modo real
                current_glucose = await sensor.get_glucose_reading()
                
                # Mesmo loop de controle da simulação
                # [implementação similar à simulação]
                
                await asyncio.sleep(600)  # 10 minutos
                
        except ImportError:
            logger.error("Adaptadores de hardware não disponíveis")
            logger.info("Execute em modo simulação ou implemente os adaptadores")
            raise
        except Exception as e:
            logger.error(f"Erro no modo real: {e}")
            raise
    
    async def _save_timestep_data(self, step: int, glucose: float, control_output):
        """Salva dados do timestep atual no banco"""
        try:
            await self.db.save_timestep({
                'step': step,
                'timestamp': self.simulator.get_current_time(),
                'glucose': glucose,
                'insulin_dose': control_output.dose,
                'dose_type': control_output.dose_type,
                'prediction': control_output.prediction,
                'confidence': control_output.confidence,
                'pid_output': control_output.pid_component,
                'ml_adjustment': control_output.ml_component
            })
        except Exception as e:
            logger.error(f"Erro ao salvar dados: {e}")
    
    async def _incremental_training(self):
        """Executa treinamento incremental do modelo ML"""
        try:
            # Obter dados recentes para treinamento
            recent_data = await self.db.get_recent_data(hours=24)
            
            if len(recent_data) > 100:  # Mínimo de dados
                self.ml_trainer.incremental_train(recent_data)
                logger.info("Treinamento incremental executado")
                
        except Exception as e:
            logger.error(f"Erro no treinamento incremental: {e}")
    
    def stop(self):
        """Para o sistema"""
        logger.info("Parando sistema APS...")
        self.running = False
    
    async def get_metrics(self, hours: int = 24) -> dict:
        """
        Calcula métricas de desempenho
        
        Args:
            hours: Período para calcular métricas
            
        Returns:
            Dicionário com métricas calculadas
        """
        try:
            data = await self.db.get_recent_data(hours=hours)
            metrics = self.metrics.calculate_all_metrics(data)
            return metrics
        except Exception as e:
            logger.error(f"Erro ao calcular métricas: {e}")
            return {}

def create_directories():
    """Cria diretórios necessários se não existirem"""
    dirs = ['logs', 'data', 'models', 'exports']
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)

async def main():
    """Função principal"""
    parser = argparse.ArgumentParser(description='Sistema de Pâncreas Artificial')
    parser.add_argument('--mode', choices=['simulation', 'real'], 
                       default='simulation', help='Modo de operação')
    parser.add_argument('--duration', type=int, default=24, 
                       help='Duração da simulação em horas')
    parser.add_argument('--acceleration', type=int, default=1, 
                       help='Aceleração temporal (1-100x)')
    parser.add_argument('--web', action='store_true', 
                       help='Iniciar interface web')
    parser.add_argument('--scenario', type=str, 
                       help='Executar cenário de teste específico')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Arquivo de configuração')
    
    args = parser.parse_args()
    
    # Criar diretórios necessários
    create_directories()
    
    try:
        # Inicializar sistema
        aps = APSSystem(config_path=args.config)
        
        if args.web:
            # Modo web com API
            app = create_app(aps)
            config = uvicorn.Config(
                app, 
                host="127.0.0.1",  # Modificado de 0.0.0.0 para 127.0.0.1
                port=8000, 
                log_level="info"
            )
            server = uvicorn.Server(config)
            await server.serve()

        elif args.scenario:
            # Executar cenário específico
            from tests.scenarios import run_scenario
            await run_scenario(aps, args.scenario)
            
        elif args.mode == 'simulation':
            # Modo simulação
            await aps.run_simulation(
                duration_hours=args.duration,
                acceleration=args.acceleration
            )
            
            # Exibir métricas finais
            metrics = await aps.get_metrics(hours=args.duration)
            print("\n=== MÉTRICAS DA SIMULAÇÃO ===")
            for key, value in metrics.items():
                print(f"{key}: {value}")
                
        elif args.mode == 'real':
            # Modo real (requer hardware)
            print("⚠️  MODO REAL - REQUER VALIDAÇÃO CLÍNICA E HARDWARE")
            print("Este modo é apenas para demonstração da arquitetura")
            await aps.run_real_mode()
            
    except KeyboardInterrupt:
        logger.info("Sistema interrompido pelo usuário")
    except Exception as e:
        logger.error(f"Erro fatal: {e}")
        sys.exit(1)
    finally:
        logger.info("Sistema APS finalizado")

if __name__ == "__main__":
    # Exibir avisos de segurança
    print("="*60)
    print("🚨 SISTEMA DE PÂNCREAS ARTIFICIAL - AVISOS IMPORTANTES 🚨")
    print("="*60)
    print("⚠️  APENAS PARA PESQUISA E SIMULAÇÃO")
    print("⚠️  NÃO USAR EM PACIENTES REAIS")
    print("⚠️  REQUER VALIDAÇÃO CLÍNICA E APROVAÇÃO REGULATÓRIA")
    print("⚠️  USO INDEVIDO PODE CAUSAR LESÕES GRAVES OU MORTE")
    print("="*60)
    print()
    
    # Executar sistema
    asyncio.run(main())