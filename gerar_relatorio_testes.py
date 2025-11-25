"""
Script para gerar relatório de testes dos modelos de deep learning.
Execute após treinar os modelos no notebook.
"""

import json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf

BASE_DATA_DIR = Path.cwd()
MODEL_OUTPUT_DIR = BASE_DATA_DIR / "modelos_treinados"
NUTRITION_CSV_PATH = BASE_DATA_DIR / "nutrition.csv"
FOOD_IMAGES_DIR = BASE_DATA_DIR / "archive (1)" / "images"

def gerar_relatorio():
    """Gera relatório completo dos testes em formato TXT."""
    
    relatorio = []
    relatorio.append("=" * 80)
    relatorio.append("RELATÓRIO DE TESTES - MODELOS DE DEEP LEARNING")
    relatorio.append("=" * 80)
    relatorio.append(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    relatorio.append("")
    
    # Informações do ambiente
    relatorio.append("-" * 80)
    relatorio.append("1. INFORMAÇÕES DO AMBIENTE")
    relatorio.append("-" * 80)
    relatorio.append(f"TensorFlow: {tf.__version__}")
    relatorio.append(f"Base de dados: {BASE_DATA_DIR}")
    relatorio.append(f"GPUs detectadas: {len(tf.config.list_physical_devices('GPU'))}")
    relatorio.append("")
    
    # Experimento 1: Regressão Nutricional
    relatorio.append("-" * 80)
    relatorio.append("2. EXPERIMENTO 1: REGRESSÃO DE CALORIAS")
    relatorio.append("-" * 80)
    
    regressor_path = MODEL_OUTPUT_DIR / "nutrition_regressor.keras"
    scaler_path = MODEL_OUTPUT_DIR / "nutrition_scaler.npy"
    
    if regressor_path.exists():
        try:
            model = tf.keras.models.load_model(regressor_path)
            relatorio.append(f"✓ Modelo carregado: {regressor_path}")
            relatorio.append(f"  Total de parâmetros: {model.count_params():,}")
            relatorio.append(f"  Arquitetura: {len(model.layers)} camadas")
        except Exception as e:
            relatorio.append(f"✗ Erro ao carregar modelo: {e}")
    else:
        relatorio.append(f"✗ Modelo não encontrado: {regressor_path}")
    
    if scaler_path.exists():
        relatorio.append(f"✓ Scaler carregado: {scaler_path}")
    else:
        relatorio.append(f"✗ Scaler não encontrado: {scaler_path}")
    
    if NUTRITION_CSV_PATH.exists():
        df = pd.read_csv(NUTRITION_CSV_PATH)
        relatorio.append(f"✓ Dataset nutricional: {len(df)} amostras")
        relatorio.append(f"  Colunas: {len(df.columns)}")
    else:
        relatorio.append(f"✗ Dataset não encontrado: {NUTRITION_CSV_PATH}")
    
    relatorio.append("")
    
    # Experimento 2: Classificação Food-101
    relatorio.append("-" * 80)
    relatorio.append("3. EXPERIMENTO 2: CLASSIFICAÇÃO FOOD-101")
    relatorio.append("-" * 80)
    
    food_model_path = MODEL_OUTPUT_DIR / "food101_classifier.keras"
    class_names_path = MODEL_OUTPUT_DIR / "food101_class_names.npy"
    
    if food_model_path.exists():
        try:
            model = tf.keras.models.load_model(food_model_path)
            relatorio.append(f"✓ Modelo carregado: {food_model_path}")
            relatorio.append(f"  Total de parâmetros: {model.count_params():,}")
            relatorio.append(f"  Arquitetura: {len(model.layers)} camadas")
        except Exception as e:
            relatorio.append(f"✗ Erro ao carregar modelo: {e}")
    else:
        relatorio.append(f"✗ Modelo não encontrado: {food_model_path}")
    
    if class_names_path.exists():
        classes = np.load(class_names_path, allow_pickle=True)
        relatorio.append(f"✓ Classes carregadas: {len(classes)} classes")
    else:
        relatorio.append(f"✗ Classes não encontradas: {class_names_path}")
    
    if FOOD_IMAGES_DIR.exists():
        classes_dirs = [d for d in FOOD_IMAGES_DIR.iterdir() if d.is_dir()]
        total_images = sum(len(list(d.glob("*.jpg"))) for d in classes_dirs)
        relatorio.append(f"✓ Dataset de imagens: {len(classes_dirs)} classes, ~{total_images:,} imagens")
    else:
        relatorio.append(f"✗ Dataset de imagens não encontrado: {FOOD_IMAGES_DIR}")
    
    relatorio.append("")
    
    # Resumo
    relatorio.append("-" * 80)
    relatorio.append("4. RESUMO")
    relatorio.append("-" * 80)
    
    modelos_ok = sum([
        regressor_path.exists(),
        food_model_path.exists()
    ])
    relatorio.append(f"Modelos treinados: {modelos_ok}/2")
    relatorio.append("")
    relatorio.append("Status dos componentes:")
    relatorio.append(f"  - Regressor nutricional: {'✓' if regressor_path.exists() else '✗'}")
    relatorio.append(f"  - Classificador Food-101: {'✓' if food_model_path.exists() else '✗'}")
    relatorio.append(f"  - Dataset nutricional: {'✓' if NUTRITION_CSV_PATH.exists() else '✗'}")
    relatorio.append(f"  - Dataset de imagens: {'✓' if FOOD_IMAGES_DIR.exists() else '✗'}")
    
    relatorio.append("")
    relatorio.append("=" * 80)
    relatorio.append("FIM DO RELATÓRIO")
    relatorio.append("=" * 80)
    
    # Salvar relatório
    output_path = BASE_DATA_DIR / "relatorio_testes.txt"
    output_path.write_text("\n".join(relatorio), encoding="utf-8")
    print(f"Relatório salvo em: {output_path}")
    return "\n".join(relatorio)

if __name__ == "__main__":
    relatorio = gerar_relatorio()
    print("\n" + relatorio)

