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

