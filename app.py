"""
API com RandomForest + HOG para reconhecimento de alimentos.
"""

import os
from pathlib import Path
from datetime import datetime
import numpy as np
from PIL import Image
import joblib
from skimage.feature import hog
from skimage import color

from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import traceback

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg"}

Path(app.config["UPLOAD_FOLDER"]).mkdir(exist_ok=True)

# ============
# CARREGAR MODELOS
# ============
MODEL_DIR = Path.cwd() / "modelos_salvos"

# Carregar modelos com tratamento de erro para não travar o processo
rf_model = None
scaler = None
label_encoder = None
try:
    print(f"Carregando modelos de: {MODEL_DIR}")
    rf_model = joblib.load(MODEL_DIR / "rf_food_classifier.joblib")
    scaler = joblib.load(MODEL_DIR / "scaler.joblib")
    label_encoder = joblib.load(MODEL_DIR / "label_encoder.joblib")
    print("[OK] Modelos carregados com sucesso")
    try:
        classes = getattr(label_encoder, 'classes_', None)
        if classes is not None:
            print(f"[OK] Label encoder com {len(classes)} classes")
            print(f"[INFO] Classes disponiveis: {list(classes)}")
    except Exception:
        pass
except Exception as e:
    print(f"[ERRO] Erro ao carregar modelos: {e}")
    traceback.print_exc()

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]


def extract_hog_features(image_path):
    """Extrai features compatíveis com o scaler carregado.

    Estratégia:
    - Consulta `scaler.n_features_in_` (se disponível) e tenta produzir features
      com esse tamanho.
    - Suporta:
      * HOG apenas
      * RGB flatten (64x64x3=12288 ou 96x96x3=27648)
      * Features combinadas (HOG + RGB) - melhor precisão
    - Tenta múltiplos formatos automaticamente e retorna o array transformado.
    """
    # Carregar imagem em RGB
    img = Image.open(image_path).convert("RGB")

    expected = None
    try:
        expected = getattr(scaler, 'n_features_in_', None)
    except Exception:
        expected = None

    # Tentar diferentes tamanhos e métodos
    img_sizes = [96, 64, 128]  # Priorizar 96x96 (novo padrão)
    
    for img_size in img_sizes:
        # 1) Features RGB normalizadas
        img_rgb = img.resize((img_size, img_size))
        arr_rgb = np.array(img_rgb).astype(np.float32) / 255.0
        features_rgb = arr_rgb.flatten()
        
        # 2) Features HOG
        img_gray = color.rgb2gray(arr_rgb)
        features_hog = hog(
            img_gray,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            visualize=False
        )
        
        # 3) Features combinadas (HOG + RGB) - melhor precisão
        features_combined = np.concatenate([features_rgb, features_hog])
        
        # Verificar qual formato corresponde ao esperado
        if expected is not None:
            if expected == features_combined.size:
                # Features combinadas (padrão melhorado)
                X = np.array([features_combined])
                return scaler.transform(X)
            elif expected == features_rgb.size:
                # Apenas RGB
                X = np.array([features_rgb])
                return scaler.transform(X)
            elif expected == features_hog.size:
                # Apenas HOG
                X = np.array([features_hog])
                return scaler.transform(X)
    
    # Se não encontrou correspondência exata, tentar features combinadas (mais comum agora)
    try:
        img_size = 96
        img_rgb = img.resize((img_size, img_size))
        arr_rgb = np.array(img_rgb).astype(np.float32) / 255.0
        features_rgb = arr_rgb.flatten()
        img_gray = color.rgb2gray(arr_rgb)
        features_hog = hog(
            img_gray,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            visualize=False
        )
        features_combined = np.concatenate([features_rgb, features_hog])
        X = np.array([features_combined])
        return scaler.transform(X)
    except Exception as e1:
        # Fallback para apenas RGB
        try:
            img_size = 96
            img_rgb = img.resize((img_size, img_size))
            arr_rgb = np.array(img_rgb).astype(np.float32) / 255.0
            features_rgb = arr_rgb.flatten()
            X = np.array([features_rgb])
            return scaler.transform(X)
        except Exception as e2:
            raise Exception(f"Erro ao extrair features: {e1}, fallback: {e2}")


def predict(image_path):
    x = extract_hog_features(image_path)
    if rf_model is None or scaler is None or label_encoder is None:
        print("[WARN] Modelos não carregados - retornando fallback")
        return "alimento_desconhecido", 0.0

    try:
        pred = rf_model.predict(x)[0]
        proba = max(rf_model.predict_proba(x)[0])
        label = label_encoder.inverse_transform([pred])[0]
        return label, float(proba)
    except Exception as e:
        print(f"Erro durante predição: {e}")
        traceback.print_exc()
        return "erro_na_predicao", 0.0


# ============
# ROTAS
# ============
@app.route("/")
def index():
    """Página principal."""
    return render_template("index.html")


@app.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "Nenhum arquivo enviado"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "Arquivo vazio"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Formato não permitido"}), 400

    filename = datetime.now().strftime("%Y%m%d_%H%M%S_") + secure_filename(file.filename)
    filepath = Path(app.config["UPLOAD_FOLDER"]) / filename
    file.save(filepath)

    print(f"[INFO] Arquivo recebido: {filename} -> salvando em {filepath}")
    alimento, confianca = predict(str(filepath))
    print(f"[INFO] Resultado da predição: {alimento} (conf: {confianca})")

    return jsonify({
        "imagem": filename,
        "alimento_reconhecido": alimento,
        "confianca": confianca
    })


@app.route("/uploads/<filename>")
def get_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    print("Servidor rodando...")
    app.run(host="0.0.0.0", port=5000, debug=True)

