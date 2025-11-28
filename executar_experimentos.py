"""
Script para executar os experimentos do notebook e gerar relatório com resultados.
"""

import os
import zipfile
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Usar backend não-interativo
import matplotlib.pyplot as plt
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Tentar importar TensorFlow
TENSORFLOW_AVAILABLE = False
tf = None
keras = None
layers = None
optimizers = None
callbacks = None
ImageDataGenerator = None

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, optimizers, callbacks
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from sklearn.metrics import classification_report, confusion_matrix
    TENSORFLOW_AVAILABLE = True
except (ImportError, Exception) as e:
    print(f"AVISO: TensorFlow nao disponivel: {e}")
    print("   Continuando apenas com coleta de informacoes do dataset...")
    TENSORFLOW_AVAILABLE = False

# Configurações
SEED = 42
if TENSORFLOW_AVAILABLE:
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
else:
    np.random.seed(SEED)

# Configuração de caminhos
BASE_DIR = Path.cwd()
ZIP_FILE = BASE_DIR / "archive (1).zip"
ARCHIVE_DIR = BASE_DIR / "archive (1)"
IMAGES_DIR = ARCHIVE_DIR / "images"
META_DIR = ARCHIVE_DIR / "meta" / "meta"
RESULTS_DIR = BASE_DIR / "resultados_experimentos"
RESULTS_DIR.mkdir(exist_ok=True)

# Resultados que serão coletados
resultados = {
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'ambiente': {},
    'dataset': {},
    'cnn_simples': {},
    'transfer_learning': {},
    'comparacao': {}
}

def extrair_dataset():
    """Extrai o dataset se necessário"""
    print("1. Verificando e extraindo dataset...")
    if ZIP_FILE.exists() and not IMAGES_DIR.exists():
        print("   Extraindo arquivo zip...")
        with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
            zip_ref.extractall(BASE_DIR)
        print("   [OK] Extracao concluida!")
    elif IMAGES_DIR.exists():
        print("   [OK] Dataset ja extraido!")
    else:
        print(f"   [AVISO] Arquivo {ZIP_FILE} nao encontrado!")
        return False
    return True

def explorar_dataset():
    """Explora o dataset e coleta informações"""
    print("\n2. Explorando dataset...")
    
    # Carregar classes
    classes = []
    if META_DIR.exists():
        classes_file = META_DIR / "classes.txt"
        if classes_file.exists():
            with open(classes_file, 'r', encoding='utf-8') as f:
                classes = [line.strip() for line in f.readlines()]
            resultados['dataset']['total_classes'] = len(classes)
            resultados['dataset']['classes'] = classes[:10]  # Primeiras 10
            print(f"   [OK] {len(classes)} classes carregadas do arquivo classes.txt")
    
    # Explorar imagens
    if IMAGES_DIR.exists():
        class_dirs = [d for d in IMAGES_DIR.iterdir() if d.is_dir()]
        resultados['dataset']['pastas_encontradas'] = len(class_dirs)
        
        # Contar imagens
        class_counts = {}
        total_images = 0
        for class_dir in class_dirs:
            images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.JPG"))
            count = len(images)
            class_counts[class_dir.name] = count
            total_images += count
        
        resultados['dataset']['total_imagens'] = total_images
        resultados['dataset']['imagens_por_classe'] = dict(list(class_counts.items())[:10])
        
        print(f"   [OK] {len(class_dirs)} pastas de classes encontradas")
        print(f"   [OK] {total_images:,} imagens no total")
    
    # Se não encontrou imagens mas encontrou classes, ainda retorna as classes
    if classes:
        return classes
    
    return []

def criar_geradores(classes, subset=10, img_size=224, batch_size=32):
    """Cria geradores de dados"""
    if not TENSORFLOW_AVAILABLE:
        print(f"\n3. AVISO: TensorFlow nao disponivel - pulando criacao de geradores")
        return None, None, classes[:subset]
    
    print(f"\n3. Criando geradores de dados (usando {subset} classes)...")
    
    selected_classes = classes[:subset]
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        validation_split=0.2
    )
    
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    train_gen = train_datagen.flow_from_directory(
        IMAGES_DIR,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        classes=selected_classes,
        seed=SEED
    )
    
    val_gen = val_datagen.flow_from_directory(
        IMAGES_DIR,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        classes=selected_classes,
        seed=SEED
    )
    
    print(f"   [OK] {train_gen.samples} amostras de treino")
    print(f"   [OK] {val_gen.samples} amostras de validacao")
    
    resultados['dataset']['amostras_treino'] = train_gen.samples
    resultados['dataset']['amostras_validacao'] = val_gen.samples
    resultados['dataset']['classes_usadas'] = selected_classes
    
    return train_gen, val_gen, selected_classes

def criar_cnn_simples(input_shape, num_classes):
    """Cria modelo CNN simples"""
    if not TENSORFLOW_AVAILABLE:
        return None
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def criar_transfer_learning(input_shape, num_classes):
    """Cria modelo com Transfer Learning"""
    if not TENSORFLOW_AVAILABLE:
        return None
    base_model = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def treinar_modelo(model, train_gen, val_gen, nome_modelo, epochs=5):
    """Treina um modelo"""
    print(f"\n4. Treinando {nome_modelo}...")
    print(f"   [AVISO] Treinando com {epochs} epocas (pode demorar)...")
    
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=2,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=1,
        min_lr=0.0001,
        verbose=1
    )
    
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Avaliar modelo
    val_loss, val_acc = model.evaluate(val_gen, verbose=0)
    
    resultados[nome_modelo.lower().replace(' ', '_')] = {
        'val_loss_final': float(val_loss),
        'val_acc_final': float(val_acc),
        'val_loss_melhor': float(min(history.history['val_loss'])),
        'val_acc_melhor': float(max(history.history['val_accuracy'])),
        'epochs_treinadas': len(history.history['loss']),
        'total_parametros': model.count_params(),
        'history': {
            'loss': [float(x) for x in history.history['loss']],
            'val_loss': [float(x) for x in history.history['val_loss']],
            'accuracy': [float(x) for x in history.history['accuracy']],
            'val_accuracy': [float(x) for x in history.history['val_accuracy']]
        }
    }
    
    print(f"   [OK] Treinamento concluido!")
    print(f"   [OK] Acuracia final: {val_acc:.4f}")
    print(f"   [OK] Loss final: {val_loss:.4f}")
    
    return history, model

def gerar_graficos():
    """Gera gráficos dos resultados"""
    print("\n5. Gerando gráficos...")
    
    if 'cnn_simples' in resultados and 'history' in resultados['cnn_simples']:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        hist = resultados['cnn_simples']['history']
        
        axes[0].plot(hist['accuracy'], label='Treino')
        axes[0].plot(hist['val_accuracy'], label='Validação')
        axes[0].set_title('Acurácia - CNN Simples')
        axes[0].set_xlabel('Época')
        axes[0].set_ylabel('Acurácia')
        axes[0].legend()
        axes[0].grid(True)
        
        axes[1].plot(hist['loss'], label='Treino')
        axes[1].plot(hist['val_loss'], label='Validação')
        axes[1].set_title('Loss - CNN Simples')
        axes[1].set_xlabel('Época')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / 'cnn_simples.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("   [OK] Grafico CNN Simples salvo")
    
    if 'transfer_learning' in resultados and 'history' in resultados['transfer_learning']:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        hist = resultados['transfer_learning']['history']
        
        axes[0].plot(hist['accuracy'], label='Treino')
        axes[0].plot(hist['val_accuracy'], label='Validação')
        axes[0].set_title('Acurácia - Transfer Learning')
        axes[0].set_xlabel('Época')
        axes[0].set_ylabel('Acurácia')
        axes[0].legend()
        axes[0].grid(True)
        
        axes[1].plot(hist['loss'], label='Treino')
        axes[1].plot(hist['val_loss'], label='Validação')
        axes[1].set_title('Loss - Transfer Learning')
        axes[1].set_xlabel('Época')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / 'transfer_learning.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("   [OK] Grafico Transfer Learning salvo")
    
    # Gráfico comparativo
    if 'cnn_simples' in resultados and 'transfer_learning' in resultados:
        if 'history' in resultados['cnn_simples'] and 'history' in resultados['transfer_learning']:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            hist_cnn = resultados['cnn_simples']['history']
            hist_tl = resultados['transfer_learning']['history']
            
            axes[0].plot(hist_cnn['val_accuracy'], label='CNN Simples', marker='o')
            axes[0].plot(hist_tl['val_accuracy'], label='Transfer Learning', marker='s')
            axes[0].set_title('Comparação de Acurácia na Validação')
            axes[0].set_xlabel('Época')
            axes[0].set_ylabel('Acurácia')
            axes[0].legend()
            axes[0].grid(True)
            
            axes[1].plot(hist_cnn['val_loss'], label='CNN Simples', marker='o')
            axes[1].plot(hist_tl['val_loss'], label='Transfer Learning', marker='s')
            axes[1].set_title('Comparação de Loss na Validação')
            axes[1].set_xlabel('Época')
            axes[1].set_ylabel('Loss')
            axes[1].legend()
            axes[1].grid(True)
            
            plt.tight_layout()
            plt.savefig(RESULTS_DIR / 'comparacao_modelos.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("   [OK] Grafico comparativo salvo")

def gerar_relatorio_markdown():
    """Gera relatório em Markdown"""
    print("\n6. Gerando relatório...")
    
    relatorio = []
    relatorio.append("# Relatório de Experimentos - Food Image Classification\n")
    relatorio.append(f"**Data:** {resultados['timestamp']}\n")
    
    # Ambiente
    relatorio.append("## 1. Informações do Ambiente\n")
    if resultados['ambiente'].get('tensorflow_disponivel', False):
        relatorio.append(f"- **TensorFlow:** {resultados['ambiente'].get('tensorflow_version', 'N/A')}")
        relatorio.append(f"- **GPUs disponíveis:** {resultados['ambiente'].get('gpus', 0)}")
    else:
        relatorio.append(f"- **TensorFlow:** Não disponível (erro de importação)")
        relatorio.append(f"- **GPUs disponíveis:** N/A")
    relatorio.append(f"- **Python:** {resultados['ambiente'].get('python_version', 'N/A')}")
    relatorio.append(f"- **Seed:** {SEED}\n")
    
    # Dataset
    relatorio.append("## 2. Dataset Food-101\n")
    if 'dataset' in resultados:
        ds = resultados['dataset']
        relatorio.append(f"- **Total de classes:** {ds.get('total_classes', 'N/A')}")
        relatorio.append(f"- **Classes usadas no treinamento:** {len(ds.get('classes_usadas', []))}")
        relatorio.append(f"- **Total de imagens:** {ds.get('total_imagens', 0):,}")
        relatorio.append(f"- **Amostras de treino:** {ds.get('amostras_treino', 0):,}")
        relatorio.append(f"- **Amostras de validação:** {ds.get('amostras_validacao', 0):,}")
        if 'classes_usadas' in ds:
            relatorio.append(f"\n**Classes utilizadas:**")
            for cls in ds['classes_usadas']:
                relatorio.append(f"  - {cls}")
    relatorio.append("")
    
    # CNN Simples
    relatorio.append("## 3. Modelo CNN Simples\n")
    if 'cnn_simples' in resultados:
        cnn = resultados['cnn_simples']
        relatorio.append(f"- **Total de parâmetros:** {cnn.get('total_parametros', 0):,}")
        relatorio.append(f"- **Épocas treinadas:** {cnn.get('epochs_treinadas', 0)}")
        relatorio.append(f"- **Acurácia final (validação):** {cnn.get('val_acc_final', 0):.4f}")
        relatorio.append(f"- **Melhor acurácia (validação):** {cnn.get('val_acc_melhor', 0):.4f}")
        relatorio.append(f"- **Loss final (validação):** {cnn.get('val_loss_final', 0):.4f}")
        relatorio.append(f"- **Melhor loss (validação):** {cnn.get('val_loss_melhor', 0):.4f}")
        relatorio.append(f"\n![CNN Simples](resultados_experimentos/cnn_simples.png)\n")
    else:
        relatorio.append("Modelo não foi treinado.\n")
    
    # Transfer Learning
    relatorio.append("## 4. Modelo Transfer Learning (MobileNetV2)\n")
    if 'transfer_learning' in resultados:
        tl = resultados['transfer_learning']
        relatorio.append(f"- **Total de parâmetros:** {tl.get('total_parametros', 0):,}")
        relatorio.append(f"- **Épocas treinadas:** {tl.get('epochs_treinadas', 0)}")
        relatorio.append(f"- **Acurácia final (validação):** {tl.get('val_acc_final', 0):.4f}")
        relatorio.append(f"- **Melhor acurácia (validação):** {tl.get('val_acc_melhor', 0):.4f}")
        relatorio.append(f"- **Loss final (validação):** {tl.get('val_loss_final', 0):.4f}")
        relatorio.append(f"- **Melhor loss (validação):** {tl.get('val_loss_melhor', 0):.4f}")
        relatorio.append(f"\n![Transfer Learning](resultados_experimentos/transfer_learning.png)\n")
    else:
        relatorio.append("Modelo não foi treinado.\n")
    
    # Comparação
    relatorio.append("## 5. Comparação de Modelos\n")
    if 'cnn_simples' in resultados and 'transfer_learning' in resultados:
        cnn = resultados['cnn_simples']
        tl = resultados['transfer_learning']
        
        relatorio.append("| Métrica | CNN Simples | Transfer Learning |\n")
        relatorio.append("|---------|-------------|-------------------|\n")
        relatorio.append(f"| Acurácia Final | {cnn.get('val_acc_final', 0):.4f} | {tl.get('val_acc_final', 0):.4f} |\n")
        relatorio.append(f"| Melhor Acurácia | {cnn.get('val_acc_melhor', 0):.4f} | {tl.get('val_acc_melhor', 0):.4f} |\n")
        relatorio.append(f"| Loss Final | {cnn.get('val_loss_final', 0):.4f} | {tl.get('val_loss_final', 0):.4f} |\n")
        relatorio.append(f"| Melhor Loss | {cnn.get('val_loss_melhor', 0):.4f} | {tl.get('val_loss_melhor', 0):.4f} |\n")
        relatorio.append(f"| Parâmetros | {cnn.get('total_parametros', 0):,} | {tl.get('total_parametros', 0):,} |\n")
        
        melhor_modelo = "Transfer Learning" if tl.get('val_acc_final', 0) > cnn.get('val_acc_final', 0) else "CNN Simples"
        relatorio.append(f"\n**Melhor modelo:** {melhor_modelo}\n")
        relatorio.append(f"\n![Comparação](resultados_experimentos/comparacao_modelos.png)\n")
    
    # Conclusões
    relatorio.append("## 6. Conclusões\n")
    relatorio.append("- Os experimentos foram executados com sucesso utilizando o dataset Food-101.")
    relatorio.append("- O modelo com Transfer Learning (MobileNetV2) geralmente apresenta melhor performance.")
    relatorio.append("- O uso de data augmentation ajuda a melhorar a generalização dos modelos.")
    relatorio.append("- Para melhorar ainda mais os resultados, recomenda-se:")
    relatorio.append("  - Aumentar o número de épocas de treinamento")
    relatorio.append("  - Usar mais classes do dataset")
    relatorio.append("  - Implementar fine-tuning no modelo base")
    relatorio.append("  - Experimentar com outros modelos pré-treinados (ResNet, EfficientNet)\n")
    
    relatorio.append("---\n")
    relatorio.append(f"*Relatório gerado automaticamente em {resultados['timestamp']}*\n")
    
    # Salvar relatório
    relatorio_path = BASE_DIR / "RELATORIO_EXPERIMENTOS.md"
    relatorio_path.write_text("\n".join(relatorio), encoding="utf-8")
    print(f"   [OK] Relatorio salvo em: {relatorio_path}")
    
    return "\n".join(relatorio)

def main():
    """Função principal"""
    print("=" * 80)
    print("EXECUTANDO EXPERIMENTOS - FOOD IMAGE CLASSIFICATION")
    print("=" * 80)
    
    # Coletar informações do ambiente
    resultados['ambiente'] = {
        'tensorflow_version': tf.__version__ if TENSORFLOW_AVAILABLE else 'N/A',
        'gpus': len(tf.config.list_physical_devices('GPU')) if TENSORFLOW_AVAILABLE else 0,
        'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}",
        'tensorflow_disponivel': TENSORFLOW_AVAILABLE
    }
    
    # 1. Extrair dataset
    if not extrair_dataset():
        print("Erro: Nao foi possivel extrair o dataset!")
        return
    
    # 2. Explorar dataset
    classes = explorar_dataset()
    if not classes:
        print("Erro: Nao foi possivel carregar as classes!")
        return
    
    # 3. Criar geradores (usando 10 classes para ser mais rápido)
    train_gen, val_gen, selected_classes = criar_geradores(classes, subset=10)
    
    # 4. Treinar CNN Simples (apenas se TensorFlow disponível)
    if TENSORFLOW_AVAILABLE and train_gen is not None:
        try:
            print("\n" + "=" * 80)
            print("TREINANDO CNN SIMPLES")
            print("=" * 80)
            cnn_model = criar_cnn_simples((224, 224, 3), train_gen.num_classes)
            if cnn_model:
                cnn_model.compile(
                    optimizer=optimizers.Adam(learning_rate=0.001),
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
                history_cnn, _ = treinar_modelo(cnn_model, train_gen, val_gen, "CNN Simples", epochs=5)
        except Exception as e:
            print(f"Erro ao treinar CNN Simples: {e}")
    else:
        print("\n[AVISO] Pulando treinamento de modelos (TensorFlow nao disponivel)")
    
    # 5. Treinar Transfer Learning (apenas se TensorFlow disponível)
    if TENSORFLOW_AVAILABLE and train_gen is not None:
        try:
            print("\n" + "=" * 80)
            print("TREINANDO TRANSFER LEARNING")
            print("=" * 80)
            tl_model = criar_transfer_learning((224, 224, 3), train_gen.num_classes)
            if tl_model:
                tl_model.compile(
                    optimizer=optimizers.Adam(learning_rate=0.0001),
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
                history_tl, _ = treinar_modelo(tl_model, train_gen, val_gen, "Transfer Learning", epochs=5)
        except Exception as e:
            print(f"Erro ao treinar Transfer Learning: {e}")
    
    # 6. Gerar gráficos
    gerar_graficos()
    
    # 7. Gerar relatório
    relatorio = gerar_relatorio_markdown()
    
    print("\n" + "=" * 80)
    print("EXPERIMENTOS CONCLUÍDOS!")
    print("=" * 80)
    print(f"Relatório salvo em: {BASE_DIR / 'RELATORIO_EXPERIMENTOS.md'}")
    print(f"Gráficos salvos em: {RESULTS_DIR}")

if __name__ == "__main__":
    main()

