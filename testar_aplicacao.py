"""
Script para testar rapidamente se a aplicação está funcionando corretamente.
"""

import sys
from pathlib import Path

def testar_importacoes():
    """Testa se todas as importações necessárias funcionam."""
    print("="*60)
    print("TESTE DE IMPORTAÇÕES")
    print("="*60)
    
    try:
        import flask
        print(f"✅ Flask {flask.__version__}")
    except ImportError as e:
        print(f"❌ Flask não encontrado: {e}")
        return False
    
    try:
        import numpy
        print(f"✅ NumPy {numpy.__version__}")
    except ImportError as e:
        print(f"❌ NumPy não encontrado: {e}")
        return False
    
    try:
        import sklearn
        print(f"✅ Scikit-learn {sklearn.__version__}")
    except ImportError as e:
        print(f"❌ Scikit-learn não encontrado: {e}")
        return False
    
    try:
        from skimage.feature import hog
        print("✅ Scikit-image")
    except ImportError as e:
        print(f"❌ Scikit-image não encontrado: {e}")
        return False
    
    try:
        import joblib
        print("✅ Joblib")
    except ImportError as e:
        print(f"❌ Joblib não encontrado: {e}")
        return False
    
    try:
        from PIL import Image
        print("✅ Pillow")
    except ImportError as e:
        print(f"❌ Pillow não encontrado: {e}")
        return False
    
    return True

def testar_banco_dados():
    """Testa se o banco de dados está funcionando."""
    print("\n" + "="*60)
    print("TESTE DE BANCO DE DADOS")
    print("="*60)
    
    try:
        from database import NutritionDB
        db = NutritionDB()
        print("✅ Banco de dados inicializado")
        
        # Testar conexão
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM refeicoes")
        count = cursor.fetchone()[0]
        print(f"✅ Conexão funcionando ({count} refeições no banco)")
        conn.close()
        return True
    except Exception as e:
        print(f"❌ Erro no banco de dados: {e}")
        return False

def testar_modelos():
    """Testa se os modelos estão disponíveis."""
    print("\n" + "="*60)
    print("TESTE DE MODELOS")
    print("="*60)
    
    MODEL_DIR = Path.cwd() / "modelos_salvos"
    
    if not MODEL_DIR.exists():
        print(f"⚠️  Diretório {MODEL_DIR} não existe")
        print("   Execute: python train_model_improved.py")
        return False
    
    modelos_necessarios = [
        "rf_food_classifier.joblib",
        "scaler.joblib",
        "label_encoder.joblib"
    ]
    
    todos_presentes = True
    for modelo in modelos_necessarios:
        caminho = MODEL_DIR / modelo
        if caminho.exists():
            tamanho = caminho.stat().st_size / (1024 * 1024)  # MB
            print(f"✅ {modelo} ({tamanho:.2f} MB)")
        else:
            print(f"❌ {modelo} não encontrado")
            todos_presentes = False
    
    if todos_presentes:
        try:
            import joblib
            rf_model = joblib.load(MODEL_DIR / "rf_food_classifier.joblib")
            scaler = joblib.load(MODEL_DIR / "scaler.joblib")
            label_encoder = joblib.load(MODEL_DIR / "label_encoder.joblib")
            print("✅ Modelos carregados com sucesso")
            return True
        except Exception as e:
            print(f"❌ Erro ao carregar modelos: {e}")
            return False
    
    return False

def testar_estrutura():
    """Testa se a estrutura de arquivos está correta."""
    print("\n" + "="*60)
    print("TESTE DE ESTRUTURA")
    print("="*60)
    
    arquivos_necessarios = [
        "app.py",
        "database.py",
        "requirements.txt",
        "Procfile",
        "runtime.txt"
    ]
    
    todos_presentes = True
    for arquivo in arquivos_necessarios:
        if Path(arquivo).exists():
            print(f"✅ {arquivo}")
        else:
            print(f"❌ {arquivo} não encontrado")
            todos_presentes = False
    
    # Verificar diretórios
    diretorios = ["templates", "uploads"]
    for diretorio in diretorios:
        if Path(diretorio).exists():
            print(f"✅ {diretorio}/")
        else:
            print(f"⚠️  {diretorio}/ não existe (será criado automaticamente)")
    
    return todos_presentes

def main():
    """Função principal."""
    print("="*60)
    print("TESTE RÁPIDO DA APLICAÇÃO")
    print("="*60)
    
    resultados = {
        "Importações": testar_importacoes(),
        "Banco de Dados": testar_banco_dados(),
        "Modelos": testar_modelos(),
        "Estrutura": testar_estrutura()
    }
    
    print("\n" + "="*60)
    print("RESUMO DOS TESTES")
    print("="*60)
    
    for teste, resultado in resultados.items():
        status = "✅ PASSOU" if resultado else "❌ FALHOU"
        print(f"{teste}: {status}")
    
    todos_passaram = all(resultados.values())
    
    print("\n" + "="*60)
    if todos_passaram:
        print("✅ TODOS OS TESTES PASSARAM!")
        print("   Aplicação pronta para uso.")
    else:
        print("⚠️  ALGUNS TESTES FALHARAM")
        print("   Verifique os erros acima e corrija antes de usar.")
    print("="*60)
    
    return 0 if todos_passaram else 1

if __name__ == "__main__":
    sys.exit(main())

