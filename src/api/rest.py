import logging
import json
import asyncio
from datetime import datetime
from enum import Enum
from typing import Optional
from fastapi import FastAPI, HTTPException, WebSocket, Response
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.websockets import WebSocketDisconnect
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles

# Configurar logger no início do arquivo
logger = logging.getLogger(__name__)

class SystemStatus(BaseModel):
    is_running: bool
    current_mode: str
    current_glucose: float = None
    system_confidence: float = None
    last_dose: float = None
    emergency_mode: bool = False

class MealRequest(BaseModel):
    carbs: float
    absorption_profile: str = "medium"
    gi_index: float = 50

class ControlCommand(BaseModel):
    action: str
    parameters: dict = {}

class WebSocketManager:
    def __init__(self):
        self.active_connections = []
        
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

class TestScenario(str, Enum):
    SENSOR_ERROR = "sensor_error"
    MEAL_CHALLENGE = "meal_challenge"
    EXERCISE = "exercise"
    STRESS = "stress"
    NIGHT = "night"
    MISSED_MEAL = "missed_meal"

class ScenarioRequest(BaseModel):
    scenario_name: TestScenario
    duration_minutes: Optional[int] = 30
    intensity: Optional[float] = 1.0

def create_app(aps_system):
    app = FastAPI(
        title="Sistema de Pâncreas Artificial",
        description="API de controle e monitoramento",
        version="1.0.0"
    )
    manager = WebSocketManager()
    
    # Variáveis de estado do aplicativo
    _current_glucose = None
    _last_dose = None
    _last_prediction = None
    _last_glucose = None
    
    # Servir arquivos estáticos
    app.mount("/static", StaticFiles(directory="src/static"), name="static")
    
    @app.get("/", response_class=HTMLResponse)
    async def root():
        """Página inicial com documentação"""
        return """
        <html>
            <head>
                <title>Sistema de Pâncreas Artificial - API</title>
                <style>
                    body {
                        font-family: Arial, sans-serif;
                        max-width: 800px;
                        margin: 0 auto;
                        padding: 20px;
                        line-height: 1.6;
                    }
                    h1 { color: #2c3e50; }
                    h2 { color: #34495e; }
                    .endpoint {
                        background: #f8f9fa;
                        padding: 10px;
                        margin: 5px 0;
                        border-radius: 4px;
                    }
                    .endpoint:hover {
                        background: #e9ecef;
                    }
                    a { color: #3498db; }
                    .method {
                        display: inline-block;
                        padding: 2px 6px;
                        border-radius: 3px;
                        font-size: 12px;
                        font-weight: bold;
                    }
                    .get { background: #61affe; color: white; }
                    .post { background: #49cc90; color: white; }
                </style>
            </head>
            <body>
                <h1>🤖 Sistema de Pâncreas Artificial</h1>
                <p>API de controle e monitoramento do sistema APS</p>
                
                <h2>📚 Documentação</h2>
                <div class="endpoint">
                    <a href="/docs">/docs</a> - Documentação Swagger (interativa)
                </div>
                <div class="endpoint">
                    <a href="/redoc">/redoc</a> - Documentação ReDoc (detalhada)
                </div>

                <h2>🔍 Endpoints Principais</h2>
                <div class="endpoint">
                    <span class="method get">GET</span>
                    <a href="/api/status">/api/status</a> - Status atual do sistema
                </div>
                <div class="endpoint">
                    <span class="method get">GET</span>
                    <a href="/api/glucose">/api/glucose</a> - Histórico de glicemia
                </div>
                <div class="endpoint">
                    <span class="method get">GET</span>
                    <a href="/api/metrics">/api/metrics</a> - Métricas de desempenho
                </div>
                <div class="endpoint">
                    <span class="method post">POST</span>
                    <a href="/docs#/default/add_meal_api_meal_post">/api/meal</a> - Adicionar refeição
                </div>
                <div class="endpoint">
                    <span class="method post">POST</span>
                    <a href="/docs#/default/control_system_api_control_post">/api/control</a> - Controlar sistema
                </div>
                
                <h2>📊 Exportação e Análise</h2>
                <div class="endpoint">
                    <span class="method get">GET</span>
                    <a href="/api/export/simulation">/api/export/simulation</a> - Exportar dados
                </div>
                
                <h2>⚡ WebSocket</h2>
                <div class="endpoint">
                    <a href="/docs#/default/websocket_endpoint_ws_get">/ws</a> - Dados em tempo real
                </div>
            </body>
        </html>
        """
    
    @app.get("/api/status")
    async def get_system_status():
        nonlocal _current_glucose, _last_dose
        """Status atual do sistema"""
        try:
            if aps_system.simulator:
                _current_glucose = aps_system.simulator.get_current_glucose()
                insulin_history = aps_system.simulator.get_insulin_history(hours=1)
                if insulin_history:
                    _last_dose = insulin_history[-1]['dose']
            
            return SystemStatus(
                is_running=aps_system.running,
                current_mode=aps_system.controller.current_mode.value,
                current_glucose=_current_glucose,
                system_confidence=aps_system.controller._calculate_system_confidence(),
                last_dose=_last_dose,
                emergency_mode=aps_system.safety_monitor.emergency_mode
            )
        except Exception as e:
            logger.error(f"Erro ao obter status: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/glucose")
    async def get_glucose_data(hours: int = 6):
        """Histórico de glicemia"""
        try:
            glucose_history = aps_system.simulator.get_glucose_history(hours=hours)
            return {
                "glucose_history": glucose_history,
                "current_time": aps_system.simulator.get_current_time(),
                "time_acceleration": aps_system.simulator.time_acceleration
            }
        except Exception as e:
            logger.error(f"Erro ao obter dados de glicose: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/metrics")
    async def get_metrics(hours: int = 24):
        """Métricas de desempenho"""
        try:
            metrics = await aps_system.get_metrics(hours=hours)
            controller_metrics = aps_system.controller.get_performance_metrics(hours=hours)
            safety_metrics = aps_system.safety_monitor.get_safety_metrics(hours=hours)
            
            return {
                "system_metrics": metrics,
                "controller_metrics": controller_metrics,
                "safety_metrics": safety_metrics,
                "period_hours": hours
            }
        except Exception as e:
            logger.error(f"Erro ao calcular métricas: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/meal")
    async def add_meal(meal: MealRequest):
        """Adicionar refeição na simulação"""
        try:
            # Verificar status do sistema
            if not aps_system.running:
                # Tentar iniciar o sistema automaticamente
                duration = 24
                acceleration = 1
                asyncio.create_task(aps_system.run_simulation(duration, acceleration))
                await asyncio.sleep(1)  # Aguardar inicialização
                
                if not aps_system.running:
                    raise HTTPException(
                        status_code=400, 
                        detail="Sistema não está rodando. Tente novamente."
                    )
            
            # Adicionar refeição
            await aps_system.simulator.consume_meal(
                carbs=meal.carbs,
                absorption_profile=meal.absorption_profile,
                gi_index=meal.gi_index
            )
            
            return {
                "message": f"Refeição adicionada: {meal.carbs}g carb",
                "timestamp": datetime.now(),
                "meal": meal.dict()
            }
        except Exception as e:
            logger.error(f"Erro ao adicionar refeição: {e}")
            if isinstance(e, HTTPException):
                raise
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/control")
    async def control_system(command: ControlCommand):
        """Controlar sistema (start/stop/pause)"""
        try:
            if command.action == "start":
                if aps_system.running:
                    return {"message": "Sistema já está rodando"}
                
                duration = command.parameters.get("duration_hours", 24)
                acceleration = command.parameters.get("acceleration", 1)
                
                # Iniciar simulação em background
                asyncio.create_task(aps_system.run_simulation(duration, acceleration))
                
                return {
                    "message": f"Sistema iniciado: {duration}h com aceleração {acceleration}x",
                    "timestamp": datetime.now()
                }
                
            elif command.action == "stop":
                aps_system.stop()
                return {
                    "message": "Sistema parado",
                    "timestamp": datetime.now()
                }
                
            elif command.action == "reset":
                aps_system.stop()
                aps_system.simulator.reset_simulation()
                return {
                    "message": "Sistema resetado",
                    "timestamp": datetime.now()
                }
                
            else:
                raise HTTPException(status_code=400, detail=f"Ação não reconhecida: {command.action}")
                
        except Exception as e:
            logger.error(f"Erro no controle do sistema: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/food-catalog")
    async def get_food_catalog():
        """Catálogo de alimentos para simulação"""
        try:
            from sim.food_catalog import get_food_catalog
            catalog = get_food_catalog()
            return {"food_catalog": catalog}
        except Exception as e:
            logger.error(f"Erro ao obter catálogo: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/inject-scenario")
    async def inject_test_scenario(scenario: ScenarioRequest):
        """
        Injetar cenário de teste
        
        Cenários disponíveis:
        - sensor_error: Simula falha no sensor
        - meal_challenge: Refeição de alto índice glicêmico
        - exercise: Atividade física
        - stress: Aumento de resistência à insulina
        - night: Período noturno
        - missed_meal: Refeição esquecida
        """
        try:
            match scenario.scenario_name:
                case TestScenario.SENSOR_ERROR:
                    aps_system.simulator.inject_sensor_error("spike", scenario.duration_minutes)
                case TestScenario.MEAL_CHALLENGE:
                    aps_system.simulator.consume_meal(60, "fast", 85)
                case TestScenario.EXERCISE:
                    aps_system.simulator.G *= (1 - 0.15 * scenario.intensity)
                case TestScenario.STRESS:
                    aps_system.simulator.insulin_sensitivity *= (1 - 0.2 * scenario.intensity)
                case TestScenario.NIGHT:
                    aps_system.simulator.basal_rate *= 0.8
                case TestScenario.MISSED_MEAL:
                    aps_system.simulator.expected_meal = True
                    
            return {
                "message": f"Cenário '{scenario.scenario_name}' injetado",
                "parameters": {
                    "duration": scenario.duration_minutes,
                    "intensity": scenario.intensity
                },
                "timestamp": datetime.now()
            }
                
        except Exception as e:
            logger.error(f"Erro ao injetar cenário: {e}")
            if isinstance(e, ValueError):
                raise HTTPException(status_code=400, detail=str(e))
            raise HTTPException(status_code=500, detail="Erro interno ao injetar cenário")
    
    @app.get("/api/export/{data_type}")
    async def export_data(data_type: str):
        """Exportar dados do sistema"""
        try:
            if data_type == "simulation":
                data = aps_system.simulator.export_simulation_data()
                filename = f"simulation_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            elif data_type == "decisions":
                filename = aps_system.controller.export_decision_history()
                return FileResponse(filename, filename=filename)
            elif data_type == "safety":
                filename = aps_system.safety_monitor.export_safety_log()
                return FileResponse(filename, filename=filename)
            else:
                raise HTTPException(status_code=400, detail="Tipo de dados inválido")
            
            return {"data": data, "filename": filename}
            
        except Exception as e:
            logger.error(f"Erro ao exportar dados: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket para dados em tempo real"""
        await manager.connect(websocket)
        try:
            while True:
                # Manter conexão viva
                data = await websocket.receive_text()
                
                # Processar comandos via WebSocket se necessário
                try:
                    command = json.loads(data)
                    if command.get("type") == "ping":
                        await websocket.send_text(json.dumps({
                            "type": "pong",
                            "timestamp": datetime.now().isoformat()
                        }))
                except json.JSONDecodeError:
                    pass  # Ignorar dados inválidos
                    
        except WebSocketDisconnect:
            manager.disconnect(websocket)
        except Exception as e:
            logger.error(f"Erro no WebSocket: {e}")
            manager.disconnect(websocket)
    
    return app