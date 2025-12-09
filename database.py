"""
Módulo para gerenciar o banco de dados SQLite da aplicação.
"""

import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
import json

DB_PATH = Path("nutrition_app.db")

class NutritionDB:
    """Classe para gerenciar o banco de dados de nutrição."""
    
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.init_db()
    
    def get_connection(self):
        """Retorna conexão com o banco de dados."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def init_db(self):
        """Inicializa as tabelas do banco de dados."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Tabela de alimentos (cache do nutrition.csv)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alimentos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nome TEXT NOT NULL UNIQUE,
                calorias REAL,
                proteinas REAL,
                carboidratos REAL,
                gorduras REAL,
                fibra REAL,
                acucar REAL,
                sodio REAL,
                dados_completos TEXT,
                criado_em TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Tabela de refeições
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS refeicoes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nome TEXT,
                imagem_path TEXT,
                alimento_reconhecido TEXT,
                confianca REAL,
                criado_em TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Tabela de itens da refeição (muitos-para-muitos)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS refeicao_itens (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                refeicao_id INTEGER NOT NULL,
                alimento_id INTEGER NOT NULL,
                quantidade REAL DEFAULT 1.0,
                FOREIGN KEY (refeicao_id) REFERENCES refeicoes(id) ON DELETE CASCADE,
                FOREIGN KEY (alimento_id) REFERENCES alimentos(id),
                UNIQUE(refeicao_id, alimento_id)
            )
        """)
        
        # Tabela de alimentos adicionados manualmente
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alimentos_manuais (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nome TEXT NOT NULL,
                calorias REAL,
                proteinas REAL,
                carboidratos REAL,
                gorduras REAL,
                criado_em TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Tabela de feedback/resultados das predições
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS resultados_predicoes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                refeicao_id INTEGER NOT NULL,
                alimento_predito TEXT NOT NULL,
                alimento_correto TEXT,
                confianca REAL,
                acertou INTEGER DEFAULT 0,
                observacoes TEXT,
                criado_em TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                atualizado_em TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (refeicao_id) REFERENCES refeicoes(id) ON DELETE CASCADE
            )
        """)
        
        conn.commit()
        conn.close()
    
    def adicionar_alimento(self, nome: str, dados_nutricionais: Dict[str, Any]) -> int:
        """Adiciona ou atualiza um alimento no banco."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        dados_json = json.dumps(dados_nutricionais, ensure_ascii=False)
        
        cursor.execute("""
            INSERT OR REPLACE INTO alimentos 
            (nome, calorias, proteinas, carboidratos, gorduras, fibra, acucar, sodio, dados_completos)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            nome,
            dados_nutricionais.get("calories"),
            dados_nutricionais.get("protein"),
            dados_nutricionais.get("carbohydrate"),
            dados_nutricionais.get("fat"),
            dados_nutricionais.get("fiber"),
            dados_nutricionais.get("sugar"),
            dados_nutricionais.get("sodium"),
            dados_json
        ))
        
        alimento_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return alimento_id
    
    def buscar_alimento(self, nome: str) -> Optional[Dict]:
        """Busca um alimento por nome (case-insensitive)."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM alimentos 
            WHERE LOWER(nome) LIKE LOWER(?)
            LIMIT 1
        """, (f"%{nome}%",))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return dict(row)
        return None
    
    def listar_alimentos(self, limite: int = 100) -> List[Dict]:
        """Lista todos os alimentos."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM alimentos ORDER BY nome LIMIT ?", (limite,))
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def criar_refeicao(
        self, 
        nome: Optional[str] = None,
        imagem_path: Optional[str] = None,
        alimento_reconhecido: Optional[str] = None,
        confianca: Optional[float] = None
    ) -> int:
        """Cria uma nova refeição."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO refeicoes (nome, imagem_path, alimento_reconhecido, confianca)
            VALUES (?, ?, ?, ?)
        """, (nome, imagem_path, alimento_reconhecido, confianca))
        
        refeicao_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return refeicao_id
    
    def adicionar_item_refeicao(self, refeicao_id: int, alimento_id: int, quantidade: float = 1.0):
        """Adiciona um alimento a uma refeição."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO refeicao_itens (refeicao_id, alimento_id, quantidade)
            VALUES (?, ?, ?)
        """, (refeicao_id, alimento_id, quantidade))
        
        conn.commit()
        conn.close()
    
    def obter_refeicao(self, refeicao_id: int) -> Optional[Dict]:
        """Obtém uma refeição com seus itens."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM refeicoes WHERE id = ?", (refeicao_id,))
        refeicao = cursor.fetchone()
        
        if not refeicao:
            conn.close()
            return None
        
        refeicao_dict = dict(refeicao)
        
        # Buscar itens da refeição
        cursor.execute("""
            SELECT a.*, ri.quantidade
            FROM refeicao_itens ri
            JOIN alimentos a ON ri.alimento_id = a.id
            WHERE ri.refeicao_id = ?
        """, (refeicao_id,))
        
        itens = [dict(row) for row in cursor.fetchall()]
        refeicao_dict["itens"] = itens
        
        # Calcular totais nutricionais
        totais = {
            "calorias": sum(item.get("calorias", 0) * item.get("quantidade", 1.0) for item in itens),
            "proteinas": sum(item.get("proteinas", 0) * item.get("quantidade", 1.0) for item in itens),
            "carboidratos": sum(item.get("carboidratos", 0) * item.get("quantidade", 1.0) for item in itens),
            "gorduras": sum(item.get("gorduras", 0) * item.get("quantidade", 1.0) for item in itens),
        }
        refeicao_dict["totais"] = totais
        
        conn.close()
        return refeicao_dict
    
    def listar_refeicoes(self, limite: int = 50) -> List[Dict]:
        """Lista todas as refeições."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT r.*, COUNT(ri.id) as num_itens
            FROM refeicoes r
            LEFT JOIN refeicao_itens ri ON r.id = ri.refeicao_id
            GROUP BY r.id
            ORDER BY r.criado_em DESC
            LIMIT ?
        """, (limite,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def adicionar_alimento_manual(
        self, 
        nome: str, 
        calorias: float,
        proteinas: float = 0.0,
        carboidratos: float = 0.0,
        gorduras: float = 0.0
    ) -> int:
        """Adiciona um alimento manualmente (quando não reconhecido na foto)."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO alimentos_manuais (nome, calorias, proteinas, carboidratos, gorduras)
            VALUES (?, ?, ?, ?, ?)
        """, (nome, calorias, proteinas, carboidratos, gorduras))
        
        alimento_id = cursor.lastrowid
        
        # Também adiciona na tabela principal de alimentos
        self.adicionar_alimento(nome, {
            "calories": calorias,
            "protein": proteinas,
            "carbohydrate": carboidratos,
            "fat": gorduras
        })
        
        conn.commit()
        conn.close()
        return alimento_id
    
    def salvar_resultado_predicao(
        self,
        refeicao_id: int,
        alimento_predito: str,
        confianca: float,
        alimento_correto: Optional[str] = None,
        acertou: Optional[bool] = None,
        observacoes: Optional[str] = None
    ) -> int:
        """Salva ou atualiza resultado de uma predição."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Verificar se já existe resultado para esta refeição
        cursor.execute("SELECT id FROM resultados_predicoes WHERE refeicao_id = ?", (refeicao_id,))
        resultado_existente = cursor.fetchone()
        
        if resultado_existente:
            # Atualizar resultado existente
            cursor.execute("""
                UPDATE resultados_predicoes
                SET alimento_correto = ?,
                    acertou = ?,
                    observacoes = ?,
                    atualizado_em = CURRENT_TIMESTAMP
                WHERE refeicao_id = ?
            """, (alimento_correto, 1 if acertou else 0, observacoes, refeicao_id))
            resultado_id = resultado_existente[0]
        else:
            # Criar novo resultado
            cursor.execute("""
                INSERT INTO resultados_predicoes 
                (refeicao_id, alimento_predito, alimento_correto, confianca, acertou, observacoes)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                refeicao_id,
                alimento_predito,
                alimento_correto,
                confianca,
                1 if acertou else 0,
                observacoes
            ))
            resultado_id = cursor.lastrowid
        
        conn.commit()
        conn.close()
        return resultado_id
    
    def calcular_acuracia(self) -> Dict[str, Any]:
        """Calcula estatísticas de acurácia das predições."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Total de predições com feedback
        cursor.execute("""
            SELECT COUNT(*) as total,
                   SUM(acertou) as acertos,
                   AVG(confianca) as confianca_media
            FROM resultados_predicoes
            WHERE alimento_correto IS NOT NULL
        """)
        stats = cursor.fetchone()
        
        total = stats[0] if stats[0] else 0
        acertos = stats[1] if stats[1] else 0
        confianca_media = stats[2] if stats[2] else 0.0
        
        acuracia = (acertos / total * 100) if total > 0 else 0.0
        
        # Estatísticas por confiança
        cursor.execute("""
            SELECT 
                CASE 
                    WHEN confianca >= 0.8 THEN 'Alta (>=80%)'
                    WHEN confianca >= 0.5 THEN 'Média (50-79%)'
                    ELSE 'Baixa (<50%)'
                END as faixa_confianca,
                COUNT(*) as total,
                SUM(acertou) as acertos,
                AVG(CASE WHEN acertou = 1 THEN 1.0 ELSE 0.0 END) * 100 as acuracia
            FROM resultados_predicoes
            WHERE alimento_correto IS NOT NULL
            GROUP BY faixa_confianca
        """)
        stats_por_confianca = [dict(row) for row in cursor.fetchall()]
        
        # Alimentos mais preditos
        cursor.execute("""
            SELECT alimento_predito, COUNT(*) as vezes_predito,
                   AVG(confianca) as confianca_media,
                   SUM(acertou) * 100.0 / COUNT(*) as acuracia
            FROM resultados_predicoes
            WHERE alimento_correto IS NOT NULL
            GROUP BY alimento_predito
            ORDER BY vezes_predito DESC
            LIMIT 10
        """)
        alimentos_mais_preditos = [dict(row) for row in cursor.fetchall()]
        
        # Últimas predições
        cursor.execute("""
            SELECT rp.*, r.imagem_path, r.criado_em
            FROM resultados_predicoes rp
            JOIN refeicoes r ON rp.refeicao_id = r.id
            WHERE rp.alimento_correto IS NOT NULL
            ORDER BY rp.atualizado_em DESC
            LIMIT 20
        """)
        ultimas_predicoes = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        
        return {
            "total_predicoes": total,
            "acertos": acertos,
            "erros": total - acertos,
            "acuracia_percentual": round(acuracia, 2),
            "confianca_media": round(confianca_media, 4),
            "stats_por_confianca": stats_por_confianca,
            "alimentos_mais_preditos": alimentos_mais_preditos,
            "ultimas_predicoes": ultimas_predicoes
        }
    
    def listar_resultados_sem_feedback(self, limite: int = 50) -> List[Dict]:
        """Lista predições que ainda não receberam feedback."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT rp.*, r.imagem_path, r.alimento_reconhecido, r.criado_em
            FROM resultados_predicoes rp
            JOIN refeicoes r ON rp.refeicao_id = r.id
            WHERE rp.alimento_correto IS NULL
            ORDER BY rp.criado_em DESC
            LIMIT ?
        """, (limite,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]

