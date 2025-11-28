"""
Script para gerar relatório de resultados dos experimentos.
Coleta informações do dataset e estrutura dos experimentos.
"""

import os
import zipfile
from pathlib import Path
from datetime import datetime
import json

BASE_DIR = Path.cwd()
ZIP_FILE = BASE_DIR / "archive (1).zip"
ARCHIVE_DIR = BASE_DIR / "archive (1)"
IMAGES_DIR = ARCHIVE_DIR / "images"
META_DIR = ARCHIVE_DIR / "meta" / "meta"
RESULTS_DIR = BASE_DIR / "resultados_experimentos"
RESULTS_DIR.mkdir(exist_ok=True)

def extrair_dataset():
    """Extrai o dataset se necessário"""
    if ZIP_FILE.exists() and not IMAGES_DIR.exists():
        print("Extraindo dataset...")
        with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
            zip_ref.extractall(BASE_DIR)
        print("[OK] Dataset extraido!")
    elif IMAGES_DIR.exists():
        print("[OK] Dataset ja extraido!")
    else:
        print(f"[AVISO] Arquivo {ZIP_FILE} nao encontrado!")
        return False
    return True

def coletar_info_dataset():
    """Coleta informações do dataset"""
    info = {}
    
    # Carregar classes
    classes = []
    if META_DIR.exists():
        classes_file = META_DIR / "classes.txt"
        if classes_file.exists():
            with open(classes_file, 'r') as f:
                classes = [line.strip() for line in f.readlines()]
            info['total_classes'] = len(classes)
            info['classes'] = classes
    
    # Contar imagens
    if IMAGES_DIR.exists():
        class_dirs = [d for d in IMAGES_DIR.iterdir() if d.is_dir()]
        info['pastas_encontradas'] = len(class_dirs)
        
        total_images = 0
        class_counts = {}
        for class_dir in class_dirs:
            images = list(class_dir.glob("*.jpg"))
            count = len(images)
            class_counts[class_dir.name] = count
            total_images += count
        
        info['total_imagens'] = total_images
        info['imagens_por_classe'] = class_counts
        info['media_imagens_por_classe'] = total_images / len(class_dirs) if class_dirs else 0
    
    return info, classes

def gerar_relatorio_markdown(dataset_info, classes):
    """Gera relatório em Markdown"""
    relatorio = []
    
    relatorio.append("# Relatório de Experimentos - Food Image Classification\n")
    relatorio.append(f"**Data de geração:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 1. Informações do Ambiente
    relatorio.append("## 1. Informações do Ambiente\n")
    relatorio.append("- **Dataset:** Food-101")
    relatorio.append("- **Fonte:** archive (1).zip")
    relatorio.append("- **Plataforma:** Jupyter Notebook Local")
    relatorio.append("- **Framework:** TensorFlow/Keras\n")
    
    # 2. Dataset
    relatorio.append("## 2. Dataset Food-101\n")
    if dataset_info:
        relatorio.append(f"- **Total de classes:** {dataset_info.get('total_classes', 0)}")
        relatorio.append(f"- **Pastas de classes encontradas:** {dataset_info.get('pastas_encontradas', 0)}")
        relatorio.append(f"- **Total de imagens:** {dataset_info.get('total_imagens', 0):,}")
        relatorio.append(f"- **Média de imagens por classe:** {dataset_info.get('media_imagens_por_classe', 0):.1f}")
        
        if 'imagens_por_classe' in dataset_info:
            relatorio.append("\n### Distribuição de Imagens (Primeiras 20 classes):\n")
            relatorio.append("| Classe | Número de Imagens |")
            relatorio.append("|--------|-------------------|")
            for i, (cls, count) in enumerate(list(dataset_info['imagens_por_classe'].items())[:20]):
                relatorio.append(f"| {cls} | {count} |")
    
    relatorio.append("\n### Classes do Dataset:\n")
    if classes:
        for i, cls in enumerate(classes, 1):
            relatorio.append(f"{i}. {cls}")
    
    # 3. Metodologia dos Experimentos
    relatorio.append("\n## 3. Metodologia dos Experimentos\n")
    relatorio.append("### 3.1 Preparação dos Dados\n")
    relatorio.append("- **Tamanho da imagem:** 224x224 pixels")
    relatorio.append("- **Batch size:** 32")
    relatorio.append("- **Divisão treino/validação:** 80%/20%")
    relatorio.append("- **Data augmentation:** Sim (rotação, translação, flip horizontal, zoom)")
    relatorio.append("- **Classes utilizadas:** 10 classes (subconjunto para treinamento rápido)")
    relatorio.append("- **Normalização:** Valores de pixel divididos por 255.0\n")
    
    relatorio.append("### 3.2 Modelos Implementados\n")
    relatorio.append("#### 3.2.1 CNN Simples\n")
    relatorio.append("Arquitetura:")
    relatorio.append("- Conv2D(32) + MaxPooling2D")
    relatorio.append("- Conv2D(64) + MaxPooling2D")
    relatorio.append("- Conv2D(128) + MaxPooling2D")
    relatorio.append("- Conv2D(128) + MaxPooling2D")
    relatorio.append("- Flatten + Dropout(0.5)")
    relatorio.append("- Dense(512) + Dense(num_classes)")
    relatorio.append("- **Optimizer:** Adam (lr=0.001)")
    relatorio.append("- **Loss:** Categorical Crossentropy\n")
    
    relatorio.append("#### 3.2.2 Transfer Learning (MobileNetV2)\n")
    relatorio.append("Arquitetura:")
    relatorio.append("- MobileNetV2 (pré-treinado no ImageNet, camadas congeladas)")
    relatorio.append("- GlobalAveragePooling2D")
    relatorio.append("- Dropout(0.2)")
    relatorio.append("- Dense(128) + Dropout(0.2)")
    relatorio.append("- Dense(num_classes)")
    relatorio.append("- **Optimizer:** Adam (lr=0.0001)")
    relatorio.append("- **Loss:** Categorical Crossentropy\n")
    
    # 4. Resultados Esperados
    relatorio.append("## 4. Resultados dos Experimentos\n")
    relatorio.append("### 4.1 Configuração de Treinamento\n")
    relatorio.append("- **Épocas:** 10 (com early stopping)")
    relatorio.append("- **Early Stopping:** Patience=3, monitor='val_loss'")
    relatorio.append("- **Reduce LR on Plateau:** Patience=2, factor=0.2\n")
    
    relatorio.append("### 4.2 Resultados Esperados\n")
    relatorio.append("Os experimentos foram configurados para executar no notebook `experimentos_local.ipynb`.")
    relatorio.append("Para obter os resultados completos, execute todas as células do notebook.\n")
    
    relatorio.append("**Resultados típicos esperados:**\n")
    relatorio.append("- **CNN Simples:**")
    relatorio.append("  - Acurácia de validação: ~60-75%")
    relatorio.append("  - Tempo de treinamento: ~30-60 minutos (CPU)")
    relatorio.append("- **Transfer Learning (MobileNetV2):**")
    relatorio.append("  - Acurácia de validação: ~80-90%")
    relatorio.append("  - Tempo de treinamento: ~20-40 minutos (CPU)\n")
    
    # 5. Como Executar
    relatorio.append("## 5. Como Executar os Experimentos\n")
    relatorio.append("1. Abra o notebook `experimentos_local.ipynb` no Jupyter")
    relatorio.append("2. Execute as células sequencialmente")
    relatorio.append("3. O notebook irá:")
    relatorio.append("   - Extrair o dataset automaticamente se necessário")
    relatorio.append("   - Explorar e visualizar o dataset")
    relatorio.append("   - Preparar os dados com data augmentation")
    relatorio.append("   - Treinar os modelos CNN Simples e Transfer Learning")
    relatorio.append("   - Gerar gráficos de evolução do treinamento")
    relatorio.append("   - Comparar os modelos")
    relatorio.append("   - Testar predições em imagens de exemplo")
    relatorio.append("   - Salvar os modelos treinados\n")
    
    # 6. Estrutura dos Arquivos
    relatorio.append("## 6. Estrutura dos Arquivos Gerados\n")
    relatorio.append("Após executar os experimentos, os seguintes arquivos serão gerados:\n")
    relatorio.append("- `modelos_salvos/cnn_simples_food101.h5` - Modelo CNN simples treinado")
    relatorio.append("- `modelos_salvos/transfer_learning_food101.h5` - Modelo Transfer Learning treinado")
    relatorio.append("- Gráficos de evolução do treinamento (exibidos no notebook)\n")
    
    # 7. Conclusões
    relatorio.append("## 7. Conclusões e Próximos Passos\n")
    relatorio.append("### Conclusões:\n")
    relatorio.append("- O dataset Food-101 contém 101 classes de alimentos com milhares de imagens")
    relatorio.append("- Transfer Learning geralmente oferece melhor performance que CNNs do zero")
    relatorio.append("- Data augmentation é essencial para melhorar a generalização")
    relatorio.append("- O uso de 10 classes permite treinamento rápido para experimentação\n")
    
    relatorio.append("### Próximos Passos para Melhorar os Resultados:\n")
    relatorio.append("1. **Aumentar número de classes:** Usar todas as 101 classes do dataset")
    relatorio.append("2. **Aumentar épocas:** Treinar por mais épocas (20-50)")
    relatorio.append("3. **Fine-tuning:** Descongelar camadas do modelo base para fine-tuning")
    relatorio.append("4. **Outros modelos:** Experimentar ResNet50, EfficientNet, etc.")
    relatorio.append("5. **Ensemble:** Combinar predições de múltiplos modelos")
    relatorio.append("6. **Análise detalhada:** Gerar matriz de confusão e relatório de classificação")
    relatorio.append("7. **Visualização:** Implementar grad-CAM para visualizar áreas de atenção\n")
    
    relatorio.append("---\n")
    relatorio.append(f"*Relatório gerado em {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    relatorio.append("*Para resultados completos, execute o notebook `experimentos_local.ipynb`*\n")
    
    return "\n".join(relatorio)

def main():
    """Função principal"""
    print("=" * 80)
    print("GERANDO RELATÓRIO DE RESULTADOS")
    print("=" * 80)
    
    # Extrair dataset se necessário
    if not extrair_dataset():
        print("Erro: Não foi possível extrair o dataset!")
        return
    
    # Coletar informações
    print("\nColetando informações do dataset...")
    dataset_info, classes = coletar_info_dataset()
    
    if not dataset_info:
        print("Erro: Não foi possível coletar informações do dataset!")
        return
    
    print(f"[OK] {dataset_info.get('total_classes', 0)} classes encontradas")
    print(f"[OK] {dataset_info.get('total_imagens', 0):,} imagens no total")
    
    # Gerar relatório
    print("\nGerando relatorio...")
    relatorio = gerar_relatorio_markdown(dataset_info, classes)
    
    # Salvar relatório
    relatorio_path = BASE_DIR / "RELATORIO_EXPERIMENTOS.md"
    relatorio_path.write_text(relatorio, encoding="utf-8")
    
    print(f"\n[OK] Relatorio salvo em: {relatorio_path}")
    print("=" * 80)
    
    # Também atualizar Artigo.md
    artigo_path = BASE_DIR / "Artigo.md"
    if artigo_path.exists():
        with open(artigo_path, 'r', encoding='utf-8') as f:
            artigo_content = f.read()
        
        # Adicionar seção de resultados se não existir
        if "## Resultados dos Experimentos" not in artigo_content:
            artigo_content += "\n\n" + "="*80 + "\n"
            artigo_content += "RESULTADOS DOS EXPERIMENTOS\n"
            artigo_content += "="*80 + "\n\n"
            artigo_content += "Para ver os resultados completos dos experimentos, consulte o arquivo:\n"
            artigo_content += "**RELATORIO_EXPERIMENTOS.md**\n\n"
            artigo_content += "Ou execute o notebook `experimentos_local.ipynb` para obter resultados em tempo real.\n"
            
            artigo_path.write_text(artigo_content, encoding="utf-8")
            print(f"[OK] Artigo.md atualizado com referencia aos resultados")

if __name__ == "__main__":
    main()

