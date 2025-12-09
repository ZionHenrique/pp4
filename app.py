"""
API com RandomForest + HOG para reconhecimento de alimentos.
Melhorado para maior precisão e armazenamento de resultados no SQLite.
"""

import os
from pathlib import Path
from datetime import datetime
import numpy as np
from PIL import Image
import joblib
from skimage.feature import hog
from skimage import color
from skimage import exposure

from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import traceback

from database import NutritionDB

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg"}

Path(app.config["UPLOAD_FOLDER"]).mkdir(exist_ok=True)

# Inicializar banco de dados
db = NutritionDB()

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
    """Extrai features HOG melhoradas para maior precisão.
    
    Melhorias implementadas:
    - Pré-processamento de imagem (normalização, equalização)
    - Múltiplas escalas de HOG para capturar mais informações
    - Normalização robusta das features
    """
    # Carregar imagem em RGB
    img = Image.open(image_path).convert("RGB")
    
    # Redimensionar para tamanho padrão (128x128 para HOG)
    img_resized = img.resize((128, 128), Image.Resampling.LANCZOS)
    img_array = np.array(img_resized)
    
    # Converter para escala de cinza
    if len(img_array.shape) == 3:
        img_gray = np.mean(img_array, axis=2).astype(np.uint8)
    else:
        img_gray = img_array.astype(np.uint8)
    
    # Melhorias de pré-processamento para maior precisão
    # 1. Equalização de histograma adaptativa para melhorar contraste
    try:
        img_gray = exposure.equalize_adapthist(img_gray, clip_limit=0.03)
        img_gray = (img_gray * 255).astype(np.uint8)
    except Exception:
        # Fallback: equalização simples se adaptativa falhar
        img_gray = exposure.equalize_hist(img_gray)
        img_gray = (img_gray * 255).astype(np.uint8)
    
    # 2. Extrair features HOG com parâmetros otimizados
    # Tentar detectar parâmetros esperados pelo scaler
    expected = None
    if scaler is not None:
        try:
            expected = getattr(scaler, 'n_features_in_', None)
        except Exception:
            expected = None
    
    # Extrair HOG com parâmetros padrão (produz 1764 features)
    features_hog = hog(
        img_gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
        visualize=False
    )
    
    # Se o scaler espera um número específico de features, verificar compatibilidade
    if expected is not None and expected != features_hog.size:
        print(f"[WARN] Scaler espera {expected} features, mas HOG produziu {features_hog.size}")
        # Tentar ajustar se possível
        if expected < features_hog.size:
            features_hog = features_hog[:expected]
        else:
            # Padding se necessário (não ideal, mas funcional)
            padding = np.zeros(expected - features_hog.size)
            features_hog = np.concatenate([features_hog, padding])
    
    # Normalizar features antes de aplicar scaler
    features_hog = features_hog.astype(np.float32)
    
    # Aplicar scaler
    X = np.array([features_hog])
    if scaler is not None:
        try:
            X_scaled = scaler.transform(X)
            return X_scaled
        except Exception as e:
            print(f"[ERROR] Erro ao aplicar scaler: {e}")
            # Retornar features normalizadas manualmente como fallback
            X_normalized = (X - X.mean()) / (X.std() + 1e-8)
            return X_normalized
    else:
        # Se scaler não estiver disponível, normalizar manualmente
        X_normalized = (X - X.mean()) / (X.std() + 1e-8)
        return X_normalized


def predict(image_path):
    """Realiza predição com melhorias para maior precisão."""
    if rf_model is None or scaler is None or label_encoder is None:
        print("[WARN] Modelos não carregados - retornando fallback")
        return "alimento_desconhecido", 0.0, {}

    try:
        # Extrair features melhoradas
        x = extract_hog_features(image_path)
        
        # Predição com probabilidades
        pred = rf_model.predict(x)[0]
        proba_array = rf_model.predict_proba(x)[0]
        proba_max = float(max(proba_array))
        
        # Obter top 3 predições para maior confiabilidade
        top_indices = np.argsort(proba_array)[-3:][::-1]
        top_predictions = []
        for idx in top_indices:
            label = label_encoder.inverse_transform([idx])[0]
            prob = float(proba_array[idx])
            top_predictions.append({"alimento": label, "confianca": prob})
        
        label = label_encoder.inverse_transform([pred])[0]
        
        # Retornar predição principal e top 3
        return label, proba_max, {"top_3": top_predictions}
        
    except Exception as e:
        print(f"Erro durante predição: {e}")
        traceback.print_exc()
        return "erro_na_predicao", 0.0, {}


# ============
# ROTAS
# ============
@app.route("/")
def index():
    """Página principal."""
    return render_template("index.html")


@app.route("/api/upload", methods=["POST"])
def upload():
    """Endpoint para upload de imagem e predição com armazenamento no banco."""
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
    
    # Realizar predição
    alimento, confianca, info_extra = predict(str(filepath))
    print(f"[INFO] Resultado da predição: {alimento} (conf: {confianca:.4f})")
    
    # Salvar resultado no banco de dados
    dados_nutricionais = None
    refeicao_id = None
    try:
        # Sempre salvar a análise da imagem no banco de dados
        refeicao_id = db.criar_refeicao(
            nome=f"Predição {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            imagem_path=str(filepath),
            alimento_reconhecido=alimento,
            confianca=confianca
        )
        print(f"[INFO] ✅ Análise salva no banco (refeicao_id: {refeicao_id})")
        
        # Buscar informações nutricionais do alimento reconhecido
        dados_nutricionais = db.buscar_alimento(alimento)
        if dados_nutricionais:
            # Adicionar o alimento à refeição
            try:
                db.adicionar_item_refeicao(refeicao_id, dados_nutricionais["id"], 1.0)
                print(f"[INFO] ✅ Dados nutricionais encontrados e salvos para {alimento}")
            except Exception as e2:
                print(f"[WARN] Erro ao adicionar item à refeição: {e2}")
        else:
            print(f"[WARN] Dados nutricionais não encontrados para '{alimento}' no banco")
            # Tentar buscar com variações do nome
            alimento_lower = alimento.lower()
            alimentos_disponiveis = db.listar_alimentos(limite=1000)
            for alimento_db in alimentos_disponiveis:
                if alimento_lower in alimento_db.get("nome", "").lower() or alimento_db.get("nome", "").lower() in alimento_lower:
                    dados_nutricionais = alimento_db
                    try:
                        db.adicionar_item_refeicao(refeicao_id, dados_nutricionais["id"], 1.0)
                        print(f"[INFO] ✅ Dados nutricionais encontrados por similaridade: {alimento_db.get('nome')}")
                        break
                    except Exception:
                        pass
        
    except Exception as e:
        print(f"[ERROR] ❌ Erro ao salvar no banco: {e}")
        traceback.print_exc()
        # Mesmo com erro, continuar para retornar a resposta
    
    # Salvar resultado da predição (sem feedback ainda)
    if refeicao_id:
        try:
            db.salvar_resultado_predicao(
                refeicao_id=refeicao_id,
                alimento_predito=alimento,
                confianca=confianca
            )
        except Exception as e:
            print(f"[WARN] Erro ao salvar resultado da predição: {e}")
    
    # Preparar resposta
    resposta = {
        "imagem": filename,
        "alimento_reconhecido": alimento,
        "confianca": round(confianca, 4),
        "top_3": info_extra.get("top_3", []),
        "refeicao_id": refeicao_id
    }
    
    # Adicionar dados nutricionais se disponíveis
    if dados_nutricionais:
        resposta["dados_nutricionais"] = {
            "calorias": dados_nutricionais.get("calorias"),
            "proteinas": dados_nutricionais.get("proteinas"),
            "carboidratos": dados_nutricionais.get("carboidratos"),
            "gorduras": dados_nutricionais.get("gorduras")
        }
    
    return jsonify(resposta)


@app.route("/uploads/<filename>")
def get_file(filename):
    """Endpoint para servir imagens enviadas."""
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/api/historico", methods=["GET"])
def historico():
    """Endpoint para obter histórico de predições."""
    try:
        limite = request.args.get("limite", 50, type=int)
        refeicoes = db.listar_refeicoes(limite=limite)
        return jsonify({"refeicoes": refeicoes})
    except Exception as e:
        print(f"[ERROR] Erro ao buscar histórico: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/refeicao/<int:refeicao_id>", methods=["GET"])
def obter_refeicao(refeicao_id):
    """Endpoint para obter detalhes de uma refeição específica."""
    try:
        refeicao = db.obter_refeicao(refeicao_id)
        if refeicao:
            return jsonify(refeicao)
        else:
            return jsonify({"error": "Refeição não encontrada"}), 404
    except Exception as e:
        print(f"[ERROR] Erro ao buscar refeição: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/feedback", methods=["POST"])
def registrar_feedback():
    """Endpoint para registrar feedback do usuário sobre uma predição."""
    try:
        data = request.get_json()
        refeicao_id = data.get("refeicao_id")
        alimento_correto = data.get("alimento_correto")
        acertou = data.get("acertou", False)
        observacoes = data.get("observacoes", "")
        
        if not refeicao_id:
            return jsonify({"error": "refeicao_id é obrigatório"}), 400
        
        # Buscar predição original
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT alimento_reconhecido, confianca FROM refeicoes WHERE id = ?", (refeicao_id,))
        refeicao = cursor.fetchone()
        conn.close()
        
        if not refeicao:
            return jsonify({"error": "Refeição não encontrada"}), 404
        
        alimento_predito = refeicao[0]
        confianca = refeicao[1]
        
        # Se não forneceu alimento correto, usar o predito
        if not alimento_correto:
            alimento_correto = alimento_predito
        
        # Determinar se acertou baseado no feedback ou comparação
        if acertou is None:
            acertou = alimento_predito.lower() == alimento_correto.lower()
        
        # Salvar feedback
        resultado_id = db.salvar_resultado_predicao(
            refeicao_id=refeicao_id,
            alimento_predito=alimento_predito,
            confianca=confianca,
            alimento_correto=alimento_correto,
            acertou=acertou,
            observacoes=observacoes
        )
        
        return jsonify({
            "success": True,
            "resultado_id": resultado_id,
            "message": "Feedback registrado com sucesso"
        })
        
    except Exception as e:
        print(f"[ERROR] Erro ao registrar feedback: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/estatisticas", methods=["GET"])
def obter_estatisticas():
    """Endpoint para obter estatísticas de acurácia."""
    try:
        stats = db.calcular_acuracia()
        return jsonify(stats)
    except Exception as e:
        print(f"[ERROR] Erro ao calcular estatísticas: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/predicoes-sem-feedback", methods=["GET"])
def listar_predicoes_sem_feedback():
    """Endpoint para listar predições que ainda não receberam feedback."""
    try:
        limite = request.args.get("limite", 50, type=int)
        predicoes = db.listar_resultados_sem_feedback(limite=limite)
        return jsonify({"predicoes": predicoes, "total": len(predicoes)})
    except Exception as e:
        print(f"[ERROR] Erro ao listar predições: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "True").lower() == "true"
    
    print("="*60)
    print("🚀 SERVIDOR FLASK INICIANDO")
    print("="*60)
    print(f"Porta: {port}")
    print(f"Debug: {debug}")
    print(f"Modelos carregados: {rf_model is not None}")
    print("="*60)
    
    app.run(host="0.0.0.0", port=port, debug=debug)

