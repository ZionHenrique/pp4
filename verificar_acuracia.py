"""
Script para verificar e exibir estatísticas de acurácia do modelo.
"""

from database import NutritionDB

def exibir_estatisticas():
    """Exibe estatísticas de acurácia do modelo."""
    print("="*60)
    print("📊 ESTATÍSTICAS DE ACURÁCIA DO MODELO")
    print("="*60)
    
    db = NutritionDB()
    stats = db.calcular_acuracia()
    
    if stats["total_predicoes"] == 0:
        print("\n⚠️  Nenhuma predição com feedback ainda.")
        print("   Faça upload de imagens pela interface web e forneça feedback")
        print("   para começar a coletar dados de acurácia.")
        return
    
    # Estatísticas gerais
    print("\n📈 ESTATÍSTICAS GERAIS")
    print("-" * 60)
    print(f"Total de Predições com Feedback: {stats['total_predicoes']}")
    print(f"✅ Acertos: {stats['acertos']}")
    print(f"❌ Erros: {stats['erros']}")
    print(f"\n🎯 ACURÁCIA GERAL: {stats['acuracia_percentual']}%")
    print(f"📊 Confiança Média: {stats['confianca_media']*100:.2f}%")
    
    # Determinar status da acurácia
    acuracia = stats['acuracia_percentual']
    if acuracia >= 70:
        status = "✅ EXCELENTE"
        cor = "\033[92m"  # Verde
    elif acuracia >= 50:
        status = "⚠️  BOM"
        cor = "\033[93m"  # Amarelo
    else:
        status = "❌ PRECISA MELHORAR"
        cor = "\033[91m"  # Vermelho
    
    print(f"\n{cor}{status}\033[0m")
    
    # Estatísticas por faixa de confiança
    if stats['stats_por_confianca']:
        print("\n📊 ACURÁCIA POR NÍVEL DE CONFIANÇA")
        print("-" * 60)
        for stat in stats['stats_por_confianca']:
            acuracia_faixa = float(stat['acuracia'])
            print(f"{stat['faixa_confianca']}:")
            print(f"  Total: {stat['total']} | Acertos: {stat['acertos']} | Acurácia: {acuracia_faixa:.2f}%")
    
    # Alimentos mais preditos
    if stats['alimentos_mais_preditos']:
        print("\n🍽️  ALIMENTOS MAIS PREDITOS")
        print("-" * 60)
        for i, alimento in enumerate(stats['alimentos_mais_preditos'][:10], 1):
            acuracia_alimento = float(alimento['acuracia'])
            confianca_media = float(alimento['confianca_media'])
            print(f"{i}. {alimento['alimento_predito']}")
            print(f"   Predições: {alimento['vezes_predito']} | Acurácia: {acuracia_alimento:.2f}% | Confiança Média: {confianca_media*100:.1f}%")
    
    # Últimas predições
    if stats['ultimas_predicoes']:
        print("\n🕐 ÚLTIMAS PREDIÇÕES COM FEEDBACK")
        print("-" * 60)
        for pred in stats['ultimas_predicoes'][:10]:
            resultado = "✅ Correto" if pred['acertou'] == 1 else "❌ Incorreto"
            print(f"ID {pred['id']}: {pred['alimento_predito']} → {pred.get('alimento_correto', 'N/A')} [{resultado}]")
            print(f"   Confiança: {pred['confianca']*100:.1f}%")
    
    print("\n" + "="*60)
    print("✅ Estatísticas atualizadas!")
    print("="*60)
    
    # Recomendações
    print("\n💡 RECOMENDAÇÕES:")
    if acuracia < 50:
        print("   - Considere retreinar o modelo com mais dados")
        print("   - Verifique se há classes desbalanceadas")
        print("   - Analise os erros mais comuns")
    elif acuracia < 70:
        print("   - O modelo está bom, mas pode melhorar")
        print("   - Continue coletando feedback dos usuários")
        print("   - Considere ajustar hiperparâmetros")
    else:
        print("   - Modelo está performando bem!")
        print("   - Continue monitorando a acurácia")
        print("   - Considere expandir o dataset para mais classes")

if __name__ == "__main__":
    exibir_estatisticas()
