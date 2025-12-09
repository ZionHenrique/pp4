"""
Script melhorado para treinar modelo de classificação de alimentos com maior acurácia.
Utiliza técnicas avançadas de pré-processamento e otimização de hiperparâmetros.
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from skimage.feature import hog
from skimage import exposure, transform
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Configurações
BASE_DIR = Path.cwd()
MODELS_DIR = BASE_DIR / "modelos_salvos"
MODELS_DIR.mkdir(exist_ok=True)

# Tentar encontrar o dataset
DATASET_PATHS = [
    BASE_DIR / "Food Classification dataset",
    BASE_DIR / "archive (1)" / "images",
    BASE_DIR / "food-101" / "images"
]

def find_dataset():
    """Encontra o diretório do dataset."""
    for path in DATASET_PATHS:
        if path.exists():
            return path
    return None

def extract_improved_hog_features(image_path, target_size=(128, 128)):
    """
    Extrai features HOG melhoradas com múltiplas técnicas de pré-processamento.
    """
    try:
        # Carregar imagem
        img = Image.open(image_path).convert("RGB")
        img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
        img_array = np.array(img_resized)
        
        # Converter para escala de cinza
        if len(img_array.shape) == 3:
            img_gray = np.mean(img_array, axis=2).astype(np.uint8)
        else:
            img_gray = img_array.astype(np.uint8)
        
        # Pré-processamento avançado
        # 1. Equalização adaptativa de histograma
        try:
            img_gray = exposure.equalize_adapthist(img_gray, clip_limit=0.03)
            img_gray = (img_gray * 255).astype(np.uint8)
        except:
            img_gray = exposure.equalize_hist(img_gray)
            img_gray = (img_gray * 255).astype(np.uint8)
        
        # 2. Extrair HOG com parâmetros otimizados
        features = hog(
            img_gray,
            orientations=9,
            pixels_per_cell=(8, 8),  # Reduzido de 16 para capturar mais detalhes
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            feature_vector=True,
            visualize=False
        )
        
        return features.astype(np.float32)
    except Exception as e:
        print(f"Erro ao processar {image_path}: {e}")
        return None

def load_dataset(dataset_path, max_samples_per_class=200, max_classes=20):
    """
    Carrega o dataset com balanceamento melhorado.
    """
    print("="*60)
    print("CARREGANDO DATASET")
    print("="*60)
    
    images = []
    labels = []
    class_counts = {}
    
    # Encontrar todas as classes
    classes = []
    for item in dataset_path.iterdir():
        if item.is_dir():
            classes.append(item.name)
    
    classes = sorted(classes)[:max_classes]  # Limitar número de classes
    print(f"Classes encontradas: {len(classes)}")
    print(f"Classes: {', '.join(classes[:10])}...")
    
    # Carregar imagens de cada classe
    for class_name in tqdm(classes, desc="Carregando classes"):
        class_path = dataset_path / class_name
        if not class_path.is_dir():
            continue
        
        # Encontrar todas as imagens
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(list(class_path.glob(ext)))
        
        # Limitar amostras por classe para balanceamento
        image_files = image_files[:max_samples_per_class]
        class_counts[class_name] = len(image_files)
        
        # Processar imagens
        for img_path in tqdm(image_files, desc=f"  {class_name}", leave=False):
            features = extract_improved_hog_features(img_path)
            if features is not None:
                images.append(features)
                labels.append(class_name)
    
    print(f"\n✅ Dataset carregado:")
    print(f"   Total de amostras: {len(images)}")
    print(f"   Número de classes: {len(set(labels))}")
    print(f"   Média de amostras por classe: {np.mean(list(class_counts.values())):.1f}")
    
    return np.array(images), np.array(labels), classes

def train_improved_model(X, y, class_names):
    """
    Treina modelo Random Forest com otimização de hiperparâmetros.
    """
    print("\n" + "="*60)
    print("TREINANDO MODELO MELHORADO")
    print("="*60)
    
    # Dividir dados
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Treino: {len(X_train)} amostras")
    print(f"Teste: {len(X_test)} amostras")
    
    # Normalizar features
    print("\nNormalizando features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Label encoder
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    # Otimização de hiperparâmetros com GridSearchCV
    print("\n🔍 Otimizando hiperparâmetros...")
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [20, 30, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2']
    }
    
    # Usar um subset menor para grid search (mais rápido)
    n_samples_grid = min(5000, len(X_train_scaled))
    indices = np.random.choice(len(X_train_scaled), n_samples_grid, replace=False)
    X_train_grid = X_train_scaled[indices]
    y_train_grid = y_train_encoded[indices]
    
    base_rf = RandomForestClassifier(random_state=42, n_jobs=-1, verbose=0)
    
    print("   Executando GridSearchCV (isso pode demorar alguns minutos)...")
    grid_search = GridSearchCV(
        base_rf, 
        param_grid, 
        cv=3, 
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train_grid, y_train_grid)
    
    print(f"\n✅ Melhores parâmetros encontrados:")
    for param, value in grid_search.best_params_.items():
        print(f"   {param}: {value}")
    print(f"   Melhor score (CV): {grid_search.best_score_:.4f}")
    
    # Treinar modelo final com todos os dados e melhores parâmetros
    print("\n🚀 Treinando modelo final com todos os dados...")
    best_params = grid_search.best_params_
    rf_model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1, verbose=1)
    rf_model.fit(X_train_scaled, y_train_encoded)
    
    # Avaliar modelo
    print("\n📊 Avaliando modelo...")
    train_pred = rf_model.predict(X_train_scaled)
    test_pred = rf_model.predict(X_test_scaled)
    
    train_acc = accuracy_score(y_train_encoded, train_pred)
    test_acc = accuracy_score(y_test_encoded, test_pred)
    
    print(f"\n✅ Resultados:")
    print(f"   Acurácia no treino: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"   Acurácia no teste: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    # Relatório de classificação
    print("\n📋 Relatório de Classificação:")
    print(classification_report(y_test_encoded, test_pred, target_names=label_encoder.classes_))
    
    # Cross-validation score
    print("\n🔄 Validação cruzada (5-fold)...")
    cv_scores = cross_val_score(rf_model, X_train_scaled, y_train_encoded, cv=5, scoring='accuracy')
    print(f"   Scores: {cv_scores}")
    print(f"   Média: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    return rf_model, scaler, label_encoder, test_acc

def save_models(model, scaler, label_encoder, class_names, accuracy):
    """Salva os modelos treinados."""
    print("\n" + "="*60)
    print("SALVANDO MODELOS")
    print("="*60)
    
    # Salvar modelo
    model_path = MODELS_DIR / "rf_food_classifier.joblib"
    joblib.dump(model, model_path)
    print(f"✅ Modelo salvo: {model_path}")
    
    # Salvar scaler
    scaler_path = MODELS_DIR / "scaler.joblib"
    joblib.dump(scaler, scaler_path)
    print(f"✅ Scaler salvo: {scaler_path}")
    
    # Salvar label encoder
    encoder_path = MODELS_DIR / "label_encoder.joblib"
    joblib.dump(label_encoder, encoder_path)
    print(f"✅ Label encoder salvo: {encoder_path}")
    
    # Salvar nomes das classes
    classes_path = MODELS_DIR / "class_names.npy"
    np.save(classes_path, class_names)
    print(f"✅ Classes salvas: {classes_path}")
    
    # Salvar informações do modelo
    info_path = MODELS_DIR / "model_info.txt"
    with open(info_path, 'w', encoding='utf-8') as f:
        f.write(f"Modelo: Random Forest Classifier\n")
        f.write(f"Acurácia no teste: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
        f.write(f"Número de classes: {len(class_names)}\n")
        f.write(f"Classes: {', '.join(class_names)}\n")
    print(f"✅ Informações salvas: {info_path}")
    
    print(f"\n✅ Todos os modelos salvos em: {MODELS_DIR}")

def main():
    """Função principal."""
    print("="*60)
    print("TREINAMENTO MELHORADO DE MODELO DE CLASSIFICAÇÃO")
    print("="*60)
    
    # Encontrar dataset
    dataset_path = find_dataset()
    if dataset_path is None:
        print("❌ Dataset não encontrado!")
        print("   Procurando em:")
        for path in DATASET_PATHS:
            print(f"   - {path}")
        return
    
    print(f"\n✅ Dataset encontrado: {dataset_path}")
    
    # Carregar dataset
    X, y, class_names = load_dataset(dataset_path, max_samples_per_class=200, max_classes=20)
    
    if len(X) == 0:
        print("❌ Nenhuma imagem foi carregada!")
        return
    
    # Treinar modelo
    model, scaler, label_encoder, accuracy = train_improved_model(X, y, class_names)
    
    # Salvar modelos
    save_models(model, scaler, label_encoder, class_names, accuracy)
    
    print("\n" + "="*60)
    print("✅ TREINAMENTO CONCLUÍDO COM SUCESSO!")
    print("="*60)
    print(f"\nAcurácia final: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Modelos prontos para uso na aplicação Flask!")

if __name__ == "__main__":
    main()

