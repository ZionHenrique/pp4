"""
Script para popular o banco de dados com dados do nutrition.csv.
Execute uma vez após criar o banco de dados.
"""

import pandas as pd
from pathlib import Path
from database import NutritionDB
import json

BASE_DATA_DIR = Path.cwd()
NUTRITION_CSV_PATH = BASE_DATA_DIR / "nutrition.csv"

def popular_banco():
    """Popula o banco de dados com dados do CSV."""
    
    if not NUTRITION_CSV_PATH.exists():
        print(f"❌ Arquivo {NUTRITION_CSV_PATH} não encontrado!")
        return
    
    print("Carregando CSV...")
    df = pd.read_csv(NUTRITION_CSV_PATH)
    df.columns = df.columns.str.strip()
    
    print(f"Encontradas {len(df)} amostras no CSV")
    
    db = NutritionDB()
    
    print("Populando banco de dados...")
    adicionados = 0
    erros = 0
    
    for idx, row in df.iterrows():
        try:
            nome = str(row.get("name", "")).strip()
            if not nome:
                continue
            
            # Preparar dados nutricionais
            dados = {}
            for col in df.columns:
                if col not in ["name", "serving_size"]:
                    try:
                        val = str(row[col])
                        # Remove unidades
                        val = val.replace("g", "").replace("mg", "").replace("kcal", "").replace("µg", "").strip()
                        # Tenta converter
                        if val and val.replace(".", "").replace("-", "").replace("e", "").replace("E", "").replace("+", "").isdigit():
                            dados[col] = float(val)
                        else:
                            dados[col] = None
                    except:
                        dados[col] = None
            
            # Adicionar ao banco
            db.adicionar_alimento(nome, dados)
            adicionados += 1
            
            if (idx + 1) % 100 == 0:
                print(f"Processados {idx + 1}/{len(df)} alimentos...")
                
        except Exception as e:
            erros += 1
            if erros <= 5:  # Mostrar apenas os primeiros 5 erros
                print(f"Erro ao processar linha {idx}: {e}")
    
    print(f"\n✅ Concluído!")
    print(f"   Alimentos adicionados: {adicionados}")
    print(f"   Erros: {erros}")

if __name__ == "__main__":
    popular_banco()

