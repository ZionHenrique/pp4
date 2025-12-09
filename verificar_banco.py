"""
Script para verificar se as análises de imagens estão sendo salvas no banco de dados.
"""

from database import NutritionDB
from pathlib import Path
import sqlite3

def verificar_banco():
    """Verifica o estado do banco de dados."""
    print("="*60)
    print("VERIFICAÇÃO DO BANCO DE DADOS")
    print("="*60)
    
    db = NutritionDB()
    conn = db.get_connection()
    cursor = conn.cursor()
    
    # Verificar tabelas
    print("\n📊 Tabelas no banco:")
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tabelas = cursor.fetchall()
    for tabela in tabelas:
        print(f"   ✅ {tabela[0]}")
    
    # Verificar refeições
    print("\n🍽️  Refeições salvas:")
    cursor.execute("SELECT COUNT(*) FROM refeicoes")
    total_refeicoes = cursor.fetchone()[0]
    print(f"   Total: {total_refeicoes}")
    
    if total_refeicoes > 0:
        cursor.execute("""
            SELECT id, nome, alimento_reconhecido, confianca, criado_em, imagem_path
            FROM refeicoes
            ORDER BY criado_em DESC
            LIMIT 10
        """)
        refeicoes = cursor.fetchall()
        print("\n   Últimas 10 análises:")
        for ref in refeicoes:
            print(f"   - ID: {ref[0]} | {ref[1]} | Alimento: {ref[2]} | Confiança: {ref[3]:.2f} | Data: {ref[4]}")
    
    # Verificar itens de refeições
    print("\n📦 Itens de refeições:")
    cursor.execute("SELECT COUNT(*) FROM refeicao_itens")
    total_itens = cursor.fetchone()[0]
    print(f"   Total: {total_itens}")
    
    # Verificar alimentos
    print("\n🥗 Alimentos cadastrados:")
    cursor.execute("SELECT COUNT(*) FROM alimentos")
    total_alimentos = cursor.fetchone()[0]
    print(f"   Total: {total_alimentos}")
    
    # Verificar resultados de predições
    print("\n📊 Resultados de predições:")
    cursor.execute("SELECT COUNT(*) FROM resultados_predicoes")
    total_resultados = cursor.fetchone()[0]
    print(f"   Total: {total_resultados}")
    
    com_feedback = 0
    if total_resultados > 0:
        cursor.execute("SELECT COUNT(*) FROM resultados_predicoes WHERE alimento_correto IS NOT NULL")
        com_feedback = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM resultados_predicoes WHERE acertou = 1")
        acertos = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM resultados_predicoes WHERE acertou = 0")
        erros = cursor.fetchone()[0]
        
        print(f"   Com feedback: {com_feedback}")
        print(f"   Acertos: {acertos}")
        print(f"   Erros: {erros}")
        
        if com_feedback > 0:
            acuracia = (acertos / com_feedback) * 100
            print(f"   Acurácia: {acuracia:.2f}%")
    
    # Estatísticas
    print("\n📈 Estatísticas:")
    if total_refeicoes > 0:
        cursor.execute("SELECT AVG(confianca) FROM refeicoes WHERE confianca IS NOT NULL")
        avg_conf = cursor.fetchone()[0]
        print(f"   Confiança média: {avg_conf:.4f}")
        
        cursor.execute("SELECT COUNT(DISTINCT alimento_reconhecido) FROM refeicoes")
        alimentos_unicos = cursor.fetchone()[0]
        print(f"   Alimentos únicos reconhecidos: {alimentos_unicos}")
    
    # Verificar integridade
    print("\n🔍 Verificando integridade:")
    cursor.execute("""
        SELECT r.id, r.alimento_reconhecido, COUNT(ri.id) as num_itens
        FROM refeicoes r
        LEFT JOIN refeicao_itens ri ON r.id = ri.refeicao_id
        GROUP BY r.id
        HAVING num_itens = 0
        LIMIT 5
    """)
    refeicoes_sem_itens = cursor.fetchall()
    if refeicoes_sem_itens:
        print(f"   ⚠️  {len(refeicoes_sem_itens)} refeições sem itens nutricionais")
        print("   (Isso é normal se o alimento não foi encontrado no banco)")
    else:
        print("   ✅ Todas as refeições têm itens associados ou não precisam")
    
    conn.close()
    
    print("\n" + "="*60)
    print("✅ VERIFICAÇÃO CONCLUÍDA")
    print("="*60)
    
    if total_refeicoes == 0:
        print("\n⚠️  Nenhuma análise foi salva ainda.")
        print("   Faça upload de uma imagem pela interface web para testar.")
    else:
        print(f"\n✅ Banco de dados funcionando corretamente!")
        print(f"   {total_refeicoes} análises salvas com sucesso.")
    
    if total_resultados > 0 and com_feedback == 0:
        print(f"\n💡 Dica: {total_resultados} predições aguardando feedback.")
        print("   Forneça feedback na interface web para calcular a acurácia do modelo.")

if __name__ == "__main__":
    verificar_banco()

