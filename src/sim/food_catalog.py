"""
Catálogo de alimentos com informações nutricionais para simulação

Baseado em:
- Tabela TACO (Tabela Brasileira de Composição de Alimentos)
- Índices glicêmicos da literatura internacional
- Perfis de absorção baseados em estudos fisiológicos
"""

import json
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class FoodItem:
    """Item de alimento com informações nutricionais"""
    name: str
    carbs_per_100g: float  # Carboidratos por 100g
    fiber_per_100g: float  # Fibras por 100g
    gi_index: float  # Índice glicêmico (0-100)
    absorption_profile: str  # 'fast', 'medium', 'slow'
    typical_serving_g: float  # Porção típica em gramas
    category: str  # Categoria do alimento

def get_food_catalog() -> Dict[str, FoodItem]:
    """
    Retorna catálogo completo de alimentos
    
    Returns:
        Dicionário com alimentos indexados por nome
    """
    foods = {
        # Cereais e grãos
        'arroz_branco': FoodItem(
            name="Arroz Branco Cozido",
            carbs_per_100g=28.0,
            fiber_per_100g=1.6,
            gi_index=73,
            absorption_profile='medium',
            typical_serving_g=150,
            category='cereais'
        ),
        'arroz_integral': FoodItem(
            name="Arroz Integral Cozido",
            carbs_per_100g=25.0,
            fiber_per_100g=2.7,
            gi_index=68,
            absorption_profile='slow',
            typical_serving_g=150,
            category='cereais'
        ),
        'pao_frances': FoodItem(
            name="Pão Francês",
            carbs_per_100g=58.0,
            fiber_per_100g=2.3,
            gi_index=95,
            absorption_profile='fast',
            typical_serving_g=50,
            category='paes'
        ),
        'pao_integral': FoodItem(
            name="Pão Integral",
            carbs_per_100g=43.0,
            fiber_per_100g=6.9,
            gi_index=74,
            absorption_profile='medium',
            typical_serving_g=50,
            category='paes'
        ),
        'macarrao': FoodItem(
            name="Macarrão Cozido",
            carbs_per_100g=25.0,
            fiber_per_100g=1.8,
            gi_index=49,
            absorption_profile='slow',
            typical_serving_g=200,
            category='cereais'
        ),
        
        # Frutas
        'banana': FoodItem(
            name="Banana Nanica",
            carbs_per_100g=26.0,
            fiber_per_100g=2.6,
            gi_index=51,
            absorption_profile='medium',
            typical_serving_g=120,
            category='frutas'
        ),
        'maca': FoodItem(
            name="Maçã com Casca",
            carbs_per_100g=14.0,
            fiber_per_100g=2.4,
            gi_index=36,
            absorption_profile='slow',
            typical_serving_g=180,
            category='frutas'
        ),
        'laranja': FoodItem(
            name="Laranja Pera",
            carbs_per_100g=9.0,
            fiber_per_100g=2.2,
            gi_index=45,
            absorption_profile='medium',
            typical_serving_g=200,
            category='frutas'
        ),
        'uva': FoodItem(
            name="Uva Comum",
            carbs_per_100g=16.0,
            fiber_per_100g=0.9,
            gi_index=59,
            absorption_profile='fast',
            typical_serving_g=100,
            category='frutas'
        ),
        
        # Leguminosas
        'feijao_preto': FoodItem(
            name="Feijão Preto Cozido",
            carbs_per_100g=14.0,
            fiber_per_100g=8.4,
            gi_index=30,
            absorption_profile='slow',
            typical_serving_g=120,
            category='leguminosas'
        ),
        'lentilha': FoodItem(
            name="Lentilha Cozida",
            carbs_per_100g=16.0,
            fiber_per_100g=7.9,
            gi_index=32,
            absorption_profile='slow',
            typical_serving_g=100,
            category='leguminosas'
        ),
        
        # Tubérculos
        'batata_inglesa': FoodItem(
            name="Batata Inglesa Cozida",
            carbs_per_100g=20.0,
            fiber_per_100g=1.3,
            gi_index=87,
            absorption_profile='fast',
            typical_serving_g=150,
            category='tuberculos'
        ),
        'batata_doce': FoodItem(
            name="Batata Doce Cozida",
            carbs_per_100g=18.4,
            fiber_per_100g=2.2,
            gi_index=70,
            absorption_profile='medium',
            typical_serving_g=150,
            category='tuberculos'
        ),
        
        # Doces e açúcares
        'acucar_cristal': FoodItem(
            name="Açúcar Cristal",
            carbs_per_100g=99.5,
            fiber_per_100g=0.0,
            gi_index=68,
            absorption_profile='fast',
            typical_serving_g=20,
            category='acucares'
        ),
        'mel': FoodItem(
            name="Mel de Abelha",
            carbs_per_100g=82.0,
            fiber_per_100g=0.2,
            gi_index=61,
            absorption_profile='fast',
            typical_serving_g=20,
            category='acucares'
        ),
        'chocolate_leite': FoodItem(
            name="Chocolate ao Leite",
            carbs_per_100g=56.0,
            fiber_per_100g=3.4,
            gi_index=49,
            absorption_profile='medium',
            typical_serving_g=30,
            category='doces'
        )
    }
    
    return foods

def calculate_meal_carbs(food_selections: List[Dict]) -> Dict:
    """
    Calcula carboidratos totais e perfil de absorção de uma refeição
    
    Args:
        food_selections: Lista de seleções [{'food': 'nome', 'amount_g': quantidade}]
        
    Returns:
        Dict com informações da refeição calculada
    """
    catalog = get_food_catalog()
    total_carbs = 0.0
    total_fiber = 0.0
    weighted_gi = 0.0
    absorption_profiles = []
    
    meal_details = []
    
    for selection in food_selections:
        food_name = selection['food']
        amount_g = selection['amount_g']
        
        if food_name not in catalog:
            continue
            
        food = catalog[food_name]
        
        # Calcular carboidratos da porção
        carbs_portion = (food.carbs_per_100g * amount_g) / 100.0
        fiber_portion = (food.fiber_per_100g * amount_g) / 100.0
        
        total_carbs += carbs_portion
        total_fiber += fiber_portion
        weighted_gi += food.gi_index * carbs_portion
        absorption_profiles.append(food.absorption_profile)
        
        meal_details.append({
            'food': food.name,
            'amount_g': amount_g,
            'carbs_g': carbs_portion,
            'fiber_g': fiber_portion,
            'gi_index': food.gi_index
        })
    
    # Calcular IG médio ponderado
    avg_gi = weighted_gi / total_carbs if total_carbs > 0 else 0
    
    # Determinar perfil de absorção predominante
    profile_counts = {}
    for profile in absorption_profiles:
        profile_counts[profile] = profile_counts.get(profile, 0) + 1
    
    dominant_profile = max(profile_counts, key=profile_counts.get) if profile_counts else 'medium'
    
    # Ajustar perfil baseado na fibra
    fiber_ratio = total_fiber / total_carbs if total_carbs > 0 else 0
    if fiber_ratio > 0.15:  # Alta fibra
        if dominant_profile == 'fast':
            dominant_profile = 'medium'
        elif dominant_profile == 'medium':
            dominant_profile = 'slow'
    
    return {
        'total_carbs': round(total_carbs, 1),
        'total_fiber': round(total_fiber, 1),
        'average_gi': round(avg_gi, 1),
        'absorption_profile': dominant_profile,
        'meal_details': meal_details,
        'fiber_carb_ratio': round(fiber_ratio, 3)
    }

def suggest_meal_portions(target_carbs: float, preferences: List[str] = None) -> List[Dict]:
    """
    Sugere porções de alimentos para atingir meta de carboidratos
    
    Args:
        target_carbs: Meta de carboidratos em gramas
        preferences: Lista de categorias preferidas
        
    Returns:
        Lista de sugestões de porções
    """
    catalog = get_food_catalog()
    
    if preferences is None:
        preferences = ['cereais', 'frutas']
    
    suggestions = []
    remaining_carbs = target_carbs
    
    # Filtrar alimentos por preferências
    filtered_foods = {
        name: food for name, food in catalog.items()
        if food.category in preferences
    }
    
    # Ordenar por densidade de carboidratos
    sorted_foods = sorted(
        filtered_foods.items(),
        key=lambda x: x[1].carbs_per_100g,
        reverse=True
    )
    
    for food_name, food in sorted_foods[:3]:  # Top 3 alimentos
        if remaining_carbs <= 0:
            break
            
        # Calcular porção necessária
        needed_amount = min(
            (remaining_carbs * 100) / food.carbs_per_100g,
            food.typical_serving_g * 1.5  # Máximo 1.5x porção típica
        )
        
        carbs_provided = (food.carbs_per_100g * needed_amount) / 100.0
        
        suggestions.append({
            'food': food_name,
            'food_name': food.name,
            'amount_g': round(needed_amount, 0),
            'carbs_provided': round(carbs_provided, 1),
            'typical_serving': food.typical_serving_g
        })
        
        remaining_carbs -= carbs_provided
    
    return suggestions