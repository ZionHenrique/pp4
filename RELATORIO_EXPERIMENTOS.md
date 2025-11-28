# Relatório de Experimentos - Food Image Classification

**Data:** 2025-11-27 20:34:07

## 1. Informações do Ambiente

- **TensorFlow:** Não disponível (erro de importação)
- **GPUs disponíveis:** N/A
- **Python:** 3.12
- **Seed:** 42

## 2. Dataset Food-101

- **Total de classes:** 101
- **Fonte:** archive (1).zip
- **Estrutura:** O dataset contém metadados com informações sobre 101 classes de alimentos
- **Classes usadas no treinamento:** 10 (configurável no notebook)
- **Total de imagens:** Dataset completo contém aproximadamente 101.000 imagens (1000 por classe)
- **Amostras de treino:** ~800 por classe (80% do total)
- **Amostras de validação:** ~200 por classe (20% do total)

### Classes do Dataset (101 classes):

As classes incluem: apple_pie, baby_back_ribs, baklava, beef_carpaccio, beef_tartare, beet_salad, beignets, bibimbap, bread_pudding, breakfast_burrito, bruschetta, caesar_salad, cannoli, caprese_salad, carrot_cake, ceviche, cheesecake, cheese_plate, chicken_curry, chicken_quesadilla, chicken_wings, chocolate_cake, chocolate_mousse, churros, clam_chowder, club_sandwich, crab_cakes, creme_brulee, croque_madame, cup_cakes, deviled_eggs, donuts, dumplings, edamame, eggs_benedict, escargots, falafel, filet_mignon, fish_and_chips, foie_gras, french_fries, french_onion_soup, french_toast, fried_calamari, fried_rice, frozen_yogurt, garlic_bread, gnocchi, greek_salad, grilled_cheese_sandwich, grilled_salmon, guacamole, gyoza, hamburger, hot_and_sour_soup, hot_dog, huevos_rancheros, hummus, ice_cream, lasagna, lobster_bisque, lobster_roll_sandwich, macaroni_and_cheese, macarons, miso_soup, mussels, nachos, omelette, onion_rings, oysters, pad_thai, paella, pancakes, panna_cotta, peking_duck, pho, pizza, pork_chop, poutine, prime_rib, pulled_pork_sandwich, ramen, ravioli, red_velvet_cake, risotto, samosa, sashimi, scallops, seaweed_salad, shrimp_and_grits, spaghetti_bolognese, spaghetti_carbonara, spring_rolls, steak, strawberry_shortcake, sushi, tacos, takoyaki, tiramisu, tuna_tartare, waffles

### Metodologia de Preparação dos Dados:

- **Tamanho da imagem:** 224x224 pixels (padrão para modelos pré-treinados)
- **Batch size:** 32
- **Divisão treino/validação:** 80%/20%
- **Data augmentation (treino):**
  - Rotação: ±20 graus
  - Translação: ±20% em largura e altura
  - Flip horizontal: Sim
  - Zoom: ±20%
- **Normalização:** Valores de pixel divididos por 255.0

## 3. Modelo CNN Simples

### Arquitetura:

- **Camada 1:** Conv2D(32 filtros, 3x3) + MaxPooling2D(2x2)
- **Camada 2:** Conv2D(64 filtros, 3x3) + MaxPooling2D(2x2)
- **Camada 3:** Conv2D(128 filtros, 3x3) + MaxPooling2D(2x2)
- **Camada 4:** Conv2D(128 filtros, 3x3) + MaxPooling2D(2x2)
- **Camada 5:** Flatten + Dropout(0.5)
- **Camada 6:** Dense(512) + Dense(num_classes, softmax)

### Configuração de Treinamento:

- **Optimizer:** Adam (learning_rate=0.001)
- **Loss:** Categorical Crossentropy
- **Métricas:** Accuracy
- **Épocas:** 10 (com early stopping)
- **Early Stopping:** Patience=3, monitor='val_loss'
- **Reduce LR on Plateau:** Patience=2, factor=0.2

### Resultados Esperados:

- **Acurácia de validação:** ~60-75% (com 10 classes)
- **Tempo de treinamento:** ~30-60 minutos (CPU) / ~10-20 minutos (GPU)
- **Total de parâmetros:** ~2-3 milhões

**Nota:** Os modelos precisam ser treinados executando o notebook `experimentos_local.ipynb`. Os resultados acima são baseados em experimentos típicos com esta arquitetura.

## 4. Modelo Transfer Learning (MobileNetV2)

### Arquitetura:

- **Base Model:** MobileNetV2 pré-treinado no ImageNet (camadas congeladas)
- **Camada 1:** GlobalAveragePooling2D
- **Camada 2:** Dropout(0.2) + Dense(128, ReLU)
- **Camada 3:** Dropout(0.2) + Dense(num_classes, softmax)

### Configuração de Treinamento:

- **Optimizer:** Adam (learning_rate=0.0001)
- **Loss:** Categorical Crossentropy
- **Métricas:** Accuracy
- **Épocas:** 10 (com early stopping)
- **Early Stopping:** Patience=3, monitor='val_loss'
- **Reduce LR on Plateau:** Patience=2, factor=0.2

### Resultados Esperados:

- **Acurácia de validação:** ~80-90% (com 10 classes)
- **Tempo de treinamento:** ~20-40 minutos (CPU) / ~5-15 minutos (GPU)
- **Total de parâmetros:** ~3-4 milhões (incluindo MobileNetV2 base)

**Nota:** Os modelos precisam ser treinados executando o notebook `experimentos_local.ipynb`. Os resultados acima são baseados em experimentos típicos com esta arquitetura.

## 5. Comparação de Modelos

### Resultados Esperados (com 10 classes):

| Métrica | CNN Simples | Transfer Learning |
|---------|-------------|-------------------|
| Acurácia Final (Validação) | ~60-75% | ~80-90% |
| Melhor Acurácia (Validação) | ~65-78% | ~82-92% |
| Loss Final (Validação) | ~0.8-1.2 | ~0.3-0.6 |
| Melhor Loss (Validação) | ~0.7-1.0 | ~0.25-0.5 |
| Parâmetros | ~2-3 milhões | ~3-4 milhões |
| Tempo de Treinamento (CPU) | ~30-60 min | ~20-40 min |
| Tempo de Treinamento (GPU) | ~10-20 min | ~5-15 min |

**Melhor modelo:** Transfer Learning (MobileNetV2) - geralmente apresenta melhor performance e convergência mais rápida.

### Análise Comparativa:

1. **Performance:** Transfer Learning supera CNN Simples devido ao conhecimento pré-treinado do ImageNet
2. **Velocidade de convergência:** Transfer Learning converge mais rapidamente
3. **Generalização:** Ambos se beneficiam de data augmentation, mas Transfer Learning tem melhor generalização
4. **Complexidade:** CNN Simples tem menos parâmetros, mas Transfer Learning é mais eficiente computacionalmente

**Nota:** Para obter resultados reais, execute o notebook `experimentos_local.ipynb` e treine os modelos.

## 6. Como Executar os Experimentos

### Pré-requisitos:

1. **Instalar dependências:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Verificar TensorFlow:**
   - TensorFlow 2.15.0 ou superior
   - Compatível com Python 3.8-3.11 (Python 3.12 pode ter problemas)

### Executando no Jupyter Notebook:

1. Abra o notebook `experimentos_local.ipynb` no Jupyter
2. Execute as células sequencialmente
3. O notebook irá:
   - Extrair o dataset automaticamente se necessário
   - Explorar e visualizar o dataset
   - Preparar os dados com data augmentation
   - Treinar os modelos CNN Simples e Transfer Learning
   - Gerar gráficos de evolução do treinamento
   - Comparar os modelos
   - Testar predições em imagens de exemplo
   - Salvar os modelos treinados em `modelos_salvos/`

### Executando via Script Python:

```bash
python executar_experimentos.py
```

**Nota:** O script atual detectou que TensorFlow não está disponível corretamente. Para treinar os modelos, é necessário:
- Corrigir a instalação do TensorFlow, ou
- Executar o notebook `experimentos_local.ipynb` diretamente

## 7. Estrutura dos Arquivos Gerados

Após executar os experimentos com sucesso, os seguintes arquivos serão gerados:

- `modelos_salvos/cnn_simples_food101.h5` - Modelo CNN simples treinado
- `modelos_salvos/transfer_learning_food101.h5` - Modelo Transfer Learning treinado
- `resultados_experimentos/cnn_simples.png` - Gráfico de evolução CNN Simples
- `resultados_experimentos/transfer_learning.png` - Gráfico de evolução Transfer Learning
- `resultados_experimentos/comparacao_modelos.png` - Gráfico comparativo

## 8. Conclusões

### Principais Descobertas:

- O dataset Food-101 contém 101 classes de alimentos com milhares de imagens de alta qualidade
- O modelo com Transfer Learning (MobileNetV2) geralmente apresenta melhor performance (~80-90% vs ~60-75%)
- O uso de data augmentation é essencial para melhorar a generalização dos modelos
- Transfer Learning oferece melhor relação custo-benefício (melhor performance com menos épocas)

### Próximos Passos para Melhorar os Resultados:

1. **Aumentar número de classes:** Usar todas as 101 classes do dataset (aumenta complexidade e tempo)
2. **Aumentar épocas:** Treinar por mais épocas (20-50) para melhor convergência
3. **Fine-tuning:** Descongelar camadas do modelo base MobileNetV2 para fine-tuning
4. **Outros modelos:** Experimentar ResNet50, EfficientNet, Vision Transformer (ViT)
5. **Ensemble:** Combinar predições de múltiplos modelos para melhor acurácia
6. **Análise detalhada:** Gerar matriz de confusão e relatório de classificação por classe
7. **Visualização:** Implementar grad-CAM para visualizar áreas de atenção nas imagens
8. **Otimização de hiperparâmetros:** Usar grid search ou bayesian optimization

---

*Relatório gerado automaticamente em 2025-11-27 20:34:07*
