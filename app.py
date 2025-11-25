"""
Aplicação web Flask para análise nutricional de imagens de comida.
"""

import os
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
import io

from database import NutritionDB

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Criar pasta de uploads
Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)

# Inicializar banco de dados
db = NutritionDB()

# Carregar modelos
BASE_DATA_DIR = Path.cwd()
MODEL_OUTPUT_DIR = BASE_DATA_DIR / "modelos_treinados"
NUTRITION_CSV_PATH = BASE_DATA_DIR / "nutrition.csv"

# Variáveis globais para modelos
food_model = None
food_class_names = None
nutrition_df = None
nutrition_scaler = None
nutrition_model = None

def load_models():
    """Carrega os modelos treinados."""
    global food_model, food_class_names, nutrition_df, nutrition_scaler, nutrition_model
    
    # Carregar modelo Food-101
    food_model_path = MODEL_OUTPUT_DIR / "food101_classifier.keras"
    if food_model_path.exists():
        try:
            food_model = tf.keras.models.load_model(food_model_path)
            print(f"✓ Modelo Food-101 carregado")
        except Exception as e:
            print(f"✗ Erro ao carregar modelo Food-101: {e}")
    
    # Carregar classes
    class_names_path = MODEL_OUTPUT_DIR / "food101_class_names.npy"
    if class_names_path.exists():
        food_class_names = np.load(class_names_path, allow_pickle=True).tolist()
        print(f"✓ {len(food_class_names)} classes carregadas")
    
    # Carregar dataset nutricional
    if NUTRITION_CSV_PATH.exists():
        nutrition_df = pd.read_csv(NUTRITION_CSV_PATH)
        nutrition_df.columns = nutrition_df.columns.str.strip()
        print(f"✓ Dataset nutricional carregado: {len(nutrition_df)} amostras")
    
    # Carregar modelo de regressão nutricional
    regressor_path = MODEL_OUTPUT_DIR / "nutrition_regressor.keras"
    if regressor_path.exists():
        try:
            nutrition_model = tf.keras.models.load_model(regressor_path)
            print(f"✓ Modelo de regressão nutricional carregado")
        except Exception as e:
            print(f"✗ Erro ao carregar modelo de regressão: {e}")
    
    # Carregar scaler
    scaler_path = MODEL_OUTPUT_DIR / "nutrition_scaler.npy"
    if scaler_path.exists():
        scaler_data = np.load(scaler_path, allow_pickle=True).item()
        from sklearn.preprocessing import StandardScaler
        nutrition_scaler = StandardScaler()
        nutrition_scaler.mean_ = scaler_data["mean"]
        nutrition_scaler.scale_ = scaler_data["scale"]
        print(f"✓ Scaler carregado")

def allowed_file(filename):
    """Verifica se o arquivo tem extensão permitida."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocessa imagem para o modelo."""
    img = tf.keras.utils.load_img(image_path, target_size=target_size)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    return img_array

def predict_food(image_path):
    """Prediz a classe de comida na imagem."""
    if food_model is None or food_class_names is None:
        return None, 0.0
    
    try:
        img_array = preprocess_image(image_path)
        predictions = food_model.predict(img_array, verbose=0)
        predicted_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_idx])
        food_name = food_class_names[predicted_idx]
        return food_name, confidence
    except Exception as e:
        print(f"Erro na predição: {e}")
        return None, 0.0

def buscar_dados_nutricionais(nome_alimento):
    """Busca dados nutricionais de um alimento."""
    # Primeiro tenta no banco de dados
    alimento = db.buscar_alimento(nome_alimento)
    if alimento:
        dados_str = alimento.get("dados_completos", "{}")
        if isinstance(dados_str, str):
            try:
                dados = json.loads(dados_str)
            except:
                dados = {}
        else:
            dados = dados_str
        if dados:
            return dados
    
    # Se não encontrar, busca no CSV
    if nutrition_df is not None:
        nome_clean = nome_alimento.lower().replace("_", " ").strip()
        # Tenta match exato primeiro
        matches = nutrition_df[
            nutrition_df["name"].str.lower().str.strip() == nome_clean
        ]
        
        # Se não encontrar, busca parcial
        if matches.empty:
            matches = nutrition_df[
                nutrition_df["name"].str.lower().str.contains(nome_clean, na=False)
            ]
        
        if not matches.empty:
            row = matches.iloc[0]
            dados = {}
            for col in nutrition_df.columns:
                if col not in ["name", "serving_size"]:
                    try:
                        val = str(row[col])
                        # Remove unidades comuns
                        val = val.replace("g", "").replace("mg", "").replace("kcal", "").replace("µg", "").strip()
                        # Tenta converter para float
                        if val and val.replace(".", "").replace("-", "").replace("e", "").replace("E", "").replace("+", "").isdigit():
                            dados[col] = float(val)
                        else:
                            dados[col] = None
                    except:
                        dados[col] = None
            
            # Salvar no banco para cache
            if dados:
                db.adicionar_alimento(nome_alimento, dados)
            
            return dados
    
    return None

@app.route('/')
def index():
    """Página principal."""
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_image():
    """Endpoint para upload de imagem e predição."""
    if 'file' not in request.files:
        return jsonify({'error': 'Nenhum arquivo enviado'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nenhum arquivo selecionado'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"
        filepath = Path(app.config['UPLOAD_FOLDER']) / filename
        file.save(filepath)
        
        # Predizer comida
        food_name, confidence = predict_food(str(filepath))
        
        resultado = {
            'imagem': filename,
            'alimento_reconhecido': food_name,
            'confianca': confidence,
            'dados_nutricionais': None
        }
        
        # Buscar dados nutricionais
        if food_name:
            dados = buscar_dados_nutricionais(food_name)
            resultado['dados_nutricionais'] = dados
            
            # Adicionar ao banco se não existir
            if dados:
                db.adicionar_alimento(food_name, dados)
        
        return jsonify(resultado)
    
    return jsonify({'error': 'Tipo de arquivo não permitido'}), 400

@app.route('/api/refeicao', methods=['POST'])
def criar_refeicao():
    """Cria uma nova refeição."""
    data = request.json
    
    refeicao_id = db.criar_refeicao(
        nome=data.get('nome'),
        imagem_path=data.get('imagem'),
        alimento_reconhecido=data.get('alimento_reconhecido'),
        confianca=data.get('confianca')
    )
    
    # Adicionar itens
    for item in data.get('itens', []):
        alimento_id = item.get('alimento_id')
        quantidade = item.get('quantidade', 1.0)
        if alimento_id:
            db.adicionar_item_refeicao(refeicao_id, alimento_id, quantidade)
        else:
            # Se não tem ID, buscar pelo nome
            nome_alimento = item.get('nome')
            if nome_alimento:
                alimento = db.buscar_alimento(nome_alimento)
                if alimento:
                    db.adicionar_item_refeicao(refeicao_id, alimento['id'], quantidade)
                else:
                    # Criar alimento se não existir
                    dados = item.get('dados_nutricionais', {})
                    alimento_id = db.adicionar_alimento(nome_alimento, dados)
                    db.adicionar_item_refeicao(refeicao_id, alimento_id, quantidade)
    
    refeicao = db.obter_refeicao(refeicao_id)
    return jsonify(refeicao)

@app.route('/api/refeicoes', methods=['GET'])
def listar_refeicoes():
    """Lista todas as refeições."""
    refeicoes = db.listar_refeicoes()
    return jsonify(refeicoes)

@app.route('/api/refeicao/<int:refeicao_id>', methods=['GET'])
def obter_refeicao(refeicao_id):
    """Obtém uma refeição específica."""
    refeicao = db.obter_refeicao(refeicao_id)
    if refeicao:
        return jsonify(refeicao)
    return jsonify({'error': 'Refeição não encontrada'}), 404

@app.route('/api/alimento/manual', methods=['POST'])
def adicionar_alimento_manual():
    """Adiciona um alimento manualmente."""
    data = request.json
    
    alimento_id = db.adicionar_alimento_manual(
        nome=data['nome'],
        calorias=data.get('calorias', 0),
        proteinas=data.get('proteinas', 0),
        carboidratos=data.get('carboidratos', 0),
        gorduras=data.get('gorduras', 0)
    )
    
    return jsonify({'alimento_id': alimento_id, 'mensagem': 'Alimento adicionado com sucesso'})

@app.route('/api/alimentos', methods=['GET'])
def listar_alimentos():
    """Lista todos os alimentos."""
    alimentos = db.listar_alimentos()
    return jsonify(alimentos)

@app.route('/api/buscar-alimento', methods=['GET'])
def buscar_alimento():
    """Busca um alimento por nome."""
    nome = request.args.get('nome', '')
    if not nome:
        return jsonify({'error': 'Nome não fornecido'}), 400
    
    alimento = db.buscar_alimento(nome)
    if alimento:
        dados = json.loads(alimento.get("dados_completos", "{}"))
        return jsonify({
            'id': alimento['id'],
            'nome': alimento['nome'],
            'dados_nutricionais': dados
        })
    
    # Tentar buscar no CSV
    dados = buscar_dados_nutricionais(nome)
    if dados:
        return jsonify({
            'nome': nome,
            'dados_nutricionais': dados
        })
    
    return jsonify({'error': 'Alimento não encontrado'}), 404

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve arquivos de upload."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    print("Carregando modelos...")
    load_models()
    print("Iniciando servidor Flask...")
    app.run(debug=True, host='0.0.0.0', port=5000)

