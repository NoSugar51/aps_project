"""
Scripts para executar cenários específicos de teste do Sistema APS

Cenários incluídos:
1. Período de adaptação (14 dias)
2. Desafio de refeição grande
3. Falha de sensor
4. Exercício simulado
5. Análise de desempenho comparativa
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta

# Adicionar src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from main import APSSystem

async def run_adaptation_scenario():
    """
    Cenário: Período de adaptação de 14 dias
    
    Simula período inicial onde o sistema aprende padrões do paciente
    e gradualmente aumenta autonomia do controle.
    """
    print("🔄 CENÁRIO: Período de Adaptação (14 dias)")
    print("=" * 50)
    
    # Inicializar sistema
    aps = APSSystem('config.yaml')
    
    # Acelerar tempo para simular 14 dias em ~30 minutos reais
    acceleration_factor = 50
    
    print(f"Aceleração temporal: {acceleration_factor}x")
    print("Simulando 14 dias de adaptação...")
    
    try:
        # Executar simulação
        await aps.run_simulation(
            duration_hours=14 * 24,  # 14 dias
            acceleration=acceleration_factor
        )
        
        # Analisar evolução da confiança
        controller_metrics = aps.controller.get_performance_metrics(hours=14*24)
        
        print("\n📊 RESULTADOS DA ADAPTAÇÃO:")
        print(f"Confiança final: {controller_metrics.get('system_confidence', 0)*100:.1f}%")
        print(f"Modo final: {controller_metrics.get('current_mode', 'unknown')}")
        print(f"Decisões totais: {controller_metrics.get('total_decisions', 0)}")
        
        # Métricas de desempenho por período
        print("\n📈 EVOLUÇÃO POR PERÍODO:")
        
        for day in [1, 3, 7, 14]:
            day_metrics = await aps.get_metrics(hours=24)  # Últimas 24h
            if day_metrics:
                print(f"Dia {day}:")
                print(f"  TIR: {day_metrics.get('time_in_range', 0)*100:.1f}%")
                print(f"  Glicemia média: {day_metrics.get('avg_glucose', 0):.1f} mg/dL")
                print(f"  Hipoglicemias: {day_metrics.get('hypoglycemia_events', 0)}")
        
        # Exportar dados para análise
        aps.controller.export_decision_history(f"adaptation_decisions_{datetime.now().strftime('%Y%m%d')}.csv")
        
    except KeyboardInterrupt:
        print("\n⏹️ Simulação interrompida pelo usuário")
    finally:
        aps.stop()

async def run_meal_challenge_scenario():
    """
    Cenário: Desafio de refeição grande
    
    Testa capacidade do sistema de lidar com refeição de 80g carboidratos
    com diferentes perfis de absorção.
    """
    print("🍕 CENÁRIO: Desafio de Refeição Grande")
    print("=" * 40)
    
    aps = APSSystem('config.yaml')
    
    try:
        # Iniciar sistema
        asyncio.create_task(aps.run_simulation(duration_hours=6, acceleration=5))
        await asyncio.sleep(2)  # Aguardar inicialização
        
        print("Sistema iniciado - aplicando desafios de refeição...")
        
        # Sequência de refeições desafiadoras
        meal_challenges = [
            {'time': 0, 'carbs': 80, 'profile': 'fast', 'name': 'Pizza + Refrigerante'},
            {'time': 120, 'carbs': 45, 'profile': 'slow', 'name': 'Massa Integral'},
            {'time': 240, 'carbs': 30, 'profile': 'medium', 'name': 'Lanche da Tarde'}
        ]
        
        for challenge in meal_challenges:
            # Aguardar tempo da refeição
            wait_time = challenge['time'] / 5  # Ajustar pela aceleração
            await asyncio.sleep(wait_time * 60)  # Converter para segundos
            
            print(f"🍽️ {challenge['name']}: {challenge['carbs']}g carb ({challenge['profile']})")
            
            aps.simulator.consume_meal(
                carbs=challenge['carbs'],
                absorption_profile=challenge['profile']
            )
            
            # Aguardar e verificar resposta
            await asyncio.sleep(30)  # 30 segundos reais
            current_glucose = aps.simulator.get_current_glucose()
            print(f"   Glicemia atual: {current_glucose:.1f} mg/dL")
        
        # Aguardar final da simulação
        await asyncio.sleep(60)  # 1 minuto real
        
        # Analisar resultados
        final_metrics = await aps.get_metrics(hours=6)
        
        print("\n📊 RESULTADOS DO DESAFIO:")
        if final_metrics:
            print(f"TIR: {final_metrics.get('time_in_range', 0)*100:.1f}%")
            print(f"Pico máximo: {max([r['value'] for r in aps.simulator.get_glucose_history()]):.1f} mg/dL")
            print(f"Hipoglicemias: {final_metrics.get('hypoglycemia_events', 0)}")
            print(f"Hiperglicemias: {final_metrics.get('hyperglycemia_events', 0)}")
        
    except Exception as e:
        print(f"Erro no cenário: {e}")
    finally:
        aps.stop()

async def run_sensor_failure_scenario():
    """
    Cenário: Falha de sensor
    
    Testa robustez do sistema quando sensor apresenta problemas.
    """
    print("📡 CENÁRIO: Falha de Sensor")
    print("=" * 30)
    
    aps = APSSystem('config.yaml')
    
    try:
        # Iniciar simulação
        asyncio.create_task(aps.run_simulation(duration_hours=4, acceleration=10))
        await asyncio.sleep(1)
        
        print("Sistema estabilizado - injetando falhas de sensor...")
        
        # Sequência de falhas
        sensor_failures = [
            {'time': 30, 'type': 'spike', 'duration': 20, 'description': 'Spike repentino'},
            {'time': 60, 'type': 'drift', 'duration': 30, 'description': 'Deriva gradual'},
            {'time': 120, 'type': 'dropout', 'duration': 15, 'description': 'Perda de sinal'},
            {'time': 180, 'type': 'noise', 'duration': 25, 'description': 'Ruído excessivo'}
        ]
        
        for failure in sensor_failures:
            # Aguardar momento da falha
            wait_minutes = failure['time'] / 10  # Ajustar pela aceleração
            await asyncio.sleep(wait_minutes * 60)
            
            print(f"⚠️ {failure['description']} por {failure['duration']} min")
            
            # Injetar falha
            aps.simulator.inject_sensor_error(failure['type'], failure['duration'])
            
            # Monitorar resposta do sistema de segurança
            await asyncio.sleep(10)  # Aguardar resposta
            
            safety_metrics = aps.safety_monitor.get_safety_metrics(hours=1)
            if safety_metrics:
                emergency_rate = safety_metrics.get('emergency_rate', 0)
                if emergency_rate > 0:
                    print(f"   ✅ Sistema detectou problema (taxa emergência: {emergency_rate*100:.1f}%)")
                else:
                    print("   ⚠️ Sistema não detectou problema")
        
        # Analisar recuperação
        await asyncio.sleep(30)  # Aguardar estabilização
        
        print("\n📊 ANÁLISE DA RECUPERAÇÃO:")
        final_safety = aps.safety_monitor.get_safety_metrics(hours=4)
        
        if final_safety:
            print(f"Alertas totais: {final_safety.get('total_safety_checks', 0)}")
            print(f"Taxa de avisos: {final_safety.get('warning_rate', 0)*100:.1f}%")
            print(f"Taxa crítica: {final_safety.get('critical_rate', 0)*100:.1f}%")
            print(f"Sistema em emergência: {'Sim' if final_safety.get('emergency_mode_active') else 'Não'}")
        
    except Exception as e:
        print(f"Erro no cenário: {e}")
    finally:
        aps.stop()

async def run_performance_comparison():
    """
    Cenário: Comparação de desempenho PID vs Híbrido
    
    Executa duas simulações idênticas comparando controlador
    PID puro vs híbrido PID+ML.
    """
    print("⚔️ CENÁRIO: PID vs Híbrido - Comparação")
    print("=" * 45)
    
    # Configurar cenário reproduzível
    np.random.seed(42)  # Para reprodutibilidade
    
    results = {}
    
    for controller_type in ['PID', 'Híbrido']:
        print(f"\n🔄 Testando controlador {controller_type}...")
        
        aps = APSSystem('config.yaml')
        
        # Configurar tipo de controlador
        if controller_type == 'PID':
            aps.controller.ml_weight = 0.0  # Desabilitar ML
        else:
            aps.controller.ml_weight = 0.3  # ML ativo
        
        try:
            # Simulação de 48h com eventos padronizados
            asyncio.create_task(aps.run_simulation(duration_hours=48, acceleration=20))
            await asyncio.sleep(2)
            
            # Eventos de teste padronizados
            test_events = [
                {'time': 8, 'action': 'meal', 'params': {'carbs': 60, 'profile': 'medium'}},
                {'time': 12, 'action': 'meal', 'params': {'carbs': 45, 'profile': 'fast'}},
                {'time': 18, 'action': 'meal', 'params': {'carbs': 75, 'profile': 'slow'}},
                {'time': 32, 'action': 'meal', 'params': {'carbs': 50, 'profile': 'medium'}},
            ]
            
            for event in test_events:
                # Aguardar momento do evento
                wait_hours = event['time'] / 20  # Ajustar pela aceleração
                await asyncio.sleep(wait_hours * 3600)
                
                if event['action'] == 'meal':
                    aps.simulator.consume_meal(**event['params'])
                    print(f"   🍽️ Refeição: {event['params']['carbs']}g carb")
            
            # Aguardar fim da simulação
            await asyncio.sleep(120)  # 2 minutos
            
            # Coletar métricas
            metrics = await aps.get_metrics(hours=48)
            controller_metrics = aps.controller.get_performance_metrics(hours=48)
            safety_metrics = aps.safety_monitor.get_safety_metrics(hours=48)
            
            results[controller_type] = {
                'tir': metrics.get('time_in_range', 0) * 100,
                'mean_glucose': metrics.get('avg_glucose', 0),
                'cv': metrics.get('coefficient_variation', 0),
                'hypo_events': metrics.get('hypoglycemia_events', 0),
                'hyper_events': metrics.get('hyperglycemia_events', 0),
                'controller_accuracy': controller_metrics.get('avg_pid_contribution', 0),
                'safety_warnings': safety_metrics.get('warning_rate', 0) * 100
            }
            
            print(f"   ✅ TIR: {results[controller_type]['tir']:.1f}%")
            print(f"   ✅ Hipoglicemias: {results[controller_type]['hypo_events']}")
            
        except Exception as e:
            print(f"   ❌ Erro: {e}")
            results[controller_type] = None
        finally:
            aps.stop()
            await asyncio.sleep(2)  # Pausa entre testes
    
    # Comparar resultados
    print("\n📊 COMPARAÇÃO DE RESULTADOS:")
    print("=" * 50)
    
    if results['PID'] and results['Híbrido']:
        metrics_comparison = [
            ('TIR (%)', 'tir', '↑'),
            ('Glicemia Média', 'mean_glucose', '120'),
            ('CV (%)', 'cv', '↓'),
            ('Hipoglicemias', 'hypo_events', '↓'),
            ('Hiperglicemias', 'hyper_events', '↓'),
            ('Avisos Segurança (%)', 'safety_warnings', '↓')
        ]
        
        for metric_name, key, direction in metrics_comparison:
            pid_val = results['PID'][key]
            hybrid_val = results['Híbrido'][key]
            
            if direction == '↑':  # Maior é melhor
                winner = 'Híbrido' if hybrid_val > pid_val else 'PID'
                performance = '✅' if hybrid_val > pid_val else '⚠️'
            elif direction == '↓':  # Menor é melhor
                winner = 'Híbrido' if hybrid_val < pid_val else 'PID'
                performance = '✅' if hybrid_val < pid_val else '⚠️'
            else:  # Próximo ao alvo
                target = float(direction)
                pid_diff = abs(pid_val - target)
                hybrid_diff = abs(hybrid_val - target)
                winner = 'Híbrido' if hybrid_diff < pid_diff else 'PID'
                performance = '✅' if hybrid_diff < pid_diff else '⚠️'
            
            print(f"{metric_name:20} | PID: {pid_val:6.1f} | Híbrido: {hybrid_val:6.1f} | {performance} {winner}")
        
        # Conclusão
        hybrid_wins = sum(1 for _, key, direction in metrics_comparison 
                         if (direction == '↑' and results['Híbrido'][key] > results['PID'][key]) or
                            (direction == '↓' and results['Híbrido'][key] < results['PID'][key]) or
                            (direction not in ['↑', '↓'] and abs(results['Híbrido'][key] - float(direction)) < abs(results['PID'][key] - float(direction))))
        
        print(f"\n🏆 RESULTADO: Híbrido venceu em {hybrid_wins}/{len(metrics_comparison)} métricas")
        
        if hybrid_wins >= len(metrics_comparison) // 2:
            print("✅ Controlador híbrido demonstrou melhor desempenho geral")
        else:
            print("⚠️ PID puro ainda competitivo - ajustar parâmetros ML")

# Função principal para executar cenários
async def main():
    """Função principal para executar cenários"""
    scenarios = {
        '1': ('Período de Adaptação (14 dias)', run_adaptation_scenario),
        '2': ('Desafio de Refeição Grande', run_meal_challenge_scenario),
        '3': ('Falha de Sensor', run_sensor_failure_scenario),
        '4': ('Comparação PID vs Híbrido', run_performance_comparison)
    }
    
    print("🧪 CENÁRIOS DE TESTE DISPONÍVEIS")
    print("=" * 40)
    
    for key, (name, _) in scenarios.items():
        print(f"{key}. {name}")
    
    print("0. Executar todos os cenários")
    print()
    
    choice = input("Escolha um cenário (0-4): ").strip()
    
    if choice == '0':
        print("🚀 Executando todos os cenários...")
        for _, (name, func) in scenarios.items():
            print(f"\n{'='*60}")
            await func()
            print(f"✅ Cenário '{name}' concluído")
            await asyncio.sleep(2)  # Pausa entre cenários
    elif choice in scenarios:
        name, func = scenarios[choice]
        print(f"🚀 Executando cenário: {name}")
        await func()
        print(f"✅ Cenário concluído")
    else:
        print("❌ Opção inválida")

if __name__ == "__main__":
    import numpy as np
    
    # Avisos de segurança
    print("⚠️" * 20)
    print("🚨 SISTEMA EXPERIMENTAL - APENAS SIMULAÇÃO")
    print("⚠️ NÃO USAR EM PACIENTES REAIS")
    print("⚠️" * 20)
    print()
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Cenários interrompidos pelo usuário")
    except Exception as e:
        print(f"\n❌ Erro nos cenários: {e}")
        import traceback
        traceback.print_exc()