# Aplicação Web de Análise Nutricional por Imagem

Sistema completo para análise nutricional de alimentos através de upload de imagens, utilizando modelos de deep learning treinados.

## 📋 Funcionalidades

- **Upload de Imagem**: Faça upload de uma foto de comida
- **Reconhecimento Automático**: O modelo Food-101 identifica o alimento na imagem
- **Informações Nutricionais**: Busca automática de dados nutricionais no banco de dados
- **Adição Manual**: Adicione alimentos que não foram reconhecidos
- **Salvamento de Refeições**: Salve refeições completas com histórico

## 🚀 Instalação

1. Instale as dependências:
```bash
pip install -r requirements.txt
```

2. Certifique-se de que os modelos estão treinados:
   - Execute o notebook `experimentos_colab.ipynb` no Google Colab
   - Baixe os modelos treinados para a pasta `modelos_treinados/`
   - Os arquivos necessários são:
     - `food101_classifier.keras`
     - `food101_class_names.npy`
     - `nutrition_regressor.keras` (opcional)
     - `nutrition_scaler.npy` (opcional)

3. Certifique-se de que o arquivo `nutrition.csv` está na raiz do projeto

4. (Opcional) Popule o banco de dados com dados do CSV:
```bash
python popular_banco.py
```
Isso acelerará as buscas de informações nutricionais.

## 🏃 Executando a Aplicação

### Windows:
```bash
run_app.bat
```

### Linux/Mac:
```bash
python app.py
```

Ou manualmente:

1. Gere o relatório de testes (opcional):
```bash
python gerar_relatorio_testes.py
```

2. Inicie o servidor Flask:
```bash
python app.py
```

3. Acesse no navegador:
```
http://localhost:5000
```

## 📁 Estrutura do Projeto

```
pp4/
├── app.py                      # Aplicação Flask principal
├── database.py                 # Gerenciamento do banco de dados SQLite
├── gerar_relatorio_testes.py   # Script para gerar relatório de testes
├── requirements.txt            # Dependências Python
├── nutrition.csv               # Dataset nutricional
├── modelos_treinados/          # Modelos treinados (criar após treinar)
│   ├── food101_classifier.keras
│   ├── food101_class_names.npy
│   ├── nutrition_regressor.keras
│   └── nutrition_scaler.npy
├── templates/
│   └── index.html             # Interface web
├── uploads/                    # Imagens enviadas (criado automaticamente)
└── nutrition_app.db           # Banco de dados SQLite (criado automaticamente)
```

## 🗄️ Banco de Dados

O banco de dados SQLite (`nutrition_app.db`) contém:

- **alimentos**: Cache de dados nutricionais
- **refeicoes**: Histórico de refeições salvas
- **refeicao_itens**: Itens de cada refeição
- **alimentos_manuais**: Alimentos adicionados manualmente

## 📊 API Endpoints

- `POST /api/upload` - Upload de imagem e predição
- `POST /api/refeicao` - Criar nova refeição
- `GET /api/refeicoes` - Listar todas as refeições
- `GET /api/refeicao/<id>` - Obter refeição específica
- `POST /api/alimento/manual` - Adicionar alimento manualmente
- `GET /api/alimentos` - Listar todos os alimentos
- `GET /api/buscar-alimento?nome=<nome>` - Buscar alimento por nome

## 🔧 Configuração

No arquivo `app.py`, você pode ajustar:

- `BASE_DATA_DIR`: Diretório base dos dados
- `MODEL_OUTPUT_DIR`: Pasta dos modelos treinados
- `NUTRITION_CSV_PATH`: Caminho do arquivo CSV nutricional
- Porta do servidor (padrão: 5000)

## 📝 Notas

- Os modelos devem ser treinados primeiro no Google Colab usando o notebook `experimentos_colab.ipynb`
- A primeira execução pode demorar para carregar os modelos
- O banco de dados é criado automaticamente na primeira execução
- As imagens enviadas são salvas na pasta `uploads/`

