# 📊 Sistema de Feedback e Cálculo de Acurácia

## ✅ Funcionalidades Implementadas

### 1. Banco de Dados de Resultados
- ✅ **Nova tabela `resultados_predicoes`** criada para armazenar todos os resultados
- ✅ **Salvamento automático** de todas as predições ao fazer upload
- ✅ **Sistema de feedback** para usuários avaliarem predições
- ✅ **Cálculo automático de acurácia** baseado no feedback

### 2. Endpoints da API

#### `/api/feedback` (POST)
Registra feedback do usuário sobre uma predição:
```json
{
  "refeicao_id": 1,
  "alimento_correto": "maçã",
  "acertou": true,
  "observacoes": "Predição correta!"
}
```

#### `/api/estatisticas` (GET)
Retorna estatísticas completas de acurácia:
- Total de predições com feedback
- Número de acertos e erros
- Acurácia percentual geral
- Confiança média
- Estatísticas por faixa de confiança
- Alimentos mais preditos
- Últimas predições com feedback

#### `/api/predicoes-sem-feedback` (GET)
Lista predições que ainda não receberam feedback do usuário.

### 3. Interface Web
- ✅ **Botões de feedback** (Correto/Incorreto) após cada predição
- ✅ **Campo para alimento correto** (se a predição estiver errada)
- ✅ **Seção de estatísticas** com visualização completa
- ✅ **Atualização automática** das estatísticas após feedback

### 4. Scripts de Verificação
- ✅ **`verificar_acuracia.py`** - Exibe estatísticas no terminal
- ✅ **`verificar_banco.py`** - Verifica estado do banco (atualizado)

## 📊 Como Funciona

### Fluxo de Uso

1. **Upload de Imagem**
   - Usuário faz upload de uma imagem
   - Modelo faz predição
   - Resultado é **automaticamente salvo** na tabela `resultados_predicoes`

2. **Feedback do Usuário**
   - Usuário vê a predição na interface
   - Clica em "✅ Correto" ou "❌ Incorreto"
   - Pode informar o alimento correto se a predição estiver errada
   - Feedback é salvo no banco

3. **Cálculo de Acurácia**
   - Sistema calcula acurácia baseado em todos os feedbacks
   - Estatísticas são atualizadas em tempo real
   - Usuário pode ver estatísticas na interface ou via script

## 🗄️ Estrutura do Banco de Dados

### Tabela `resultados_predicoes`
```sql
- id: ID único
- refeicao_id: Referência à refeição
- alimento_predito: Alimento que o modelo predisse
- alimento_correto: Alimento correto (informado pelo usuário)
- confianca: Nível de confiança da predição
- acertou: 1 se acertou, 0 se errou
- observacoes: Observações do usuário
- criado_em: Data de criação
- atualizado_em: Data de atualização
```

## 📈 Estatísticas Disponíveis

### Acurácia Geral
- Total de predições com feedback
- Número de acertos
- Número de erros
- Percentual de acurácia
- Confiança média

### Por Faixa de Confiança
- Alta (>=80%)
- Média (50-79%)
- Baixa (<50%)

### Por Alimento
- Alimentos mais preditos
- Acurácia por alimento
- Número de vezes predito

## 🚀 Como Usar

### 1. Verificar Estatísticas no Terminal
```bash
python verificar_acuracia.py
```

### 2. Ver Estatísticas na Interface Web
- Acesse a aplicação: `http://localhost:5000`
- Role até a seção "📊 Estatísticas de Acurácia"
- Clique em "Atualizar Estatísticas"

### 3. Dar Feedback
- Após fazer upload de uma imagem
- Veja a predição
- Clique em "✅ Correto" ou "❌ Incorreto"
- Se incorreto, informe o alimento correto no campo

### 4. API REST
```bash
# Obter estatísticas
curl http://localhost:5000/api/estatisticas

# Registrar feedback
curl -X POST http://localhost:5000/api/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "refeicao_id": 1,
    "alimento_correto": "maçã",
    "acertou": true
  }'
```

## 📝 Exemplo de Resposta de Estatísticas

```json
{
  "total_predicoes": 50,
  "acertos": 35,
  "erros": 15,
  "acuracia_percentual": 70.0,
  "confianca_media": 0.7523,
  "stats_por_confianca": [
    {
      "faixa_confianca": "Alta (>=80%)",
      "total": 20,
      "acertos": 18,
      "acuracia": 90.0
    }
  ],
  "alimentos_mais_preditos": [
    {
      "alimento_predito": "maçã",
      "vezes_predito": 10,
      "confianca_media": 0.85,
      "acuracia": 80.0
    }
  ]
}
```

## ⚠️ Notas Importantes

1. **Feedback é Opcional**: O sistema funciona mesmo sem feedback, mas a acurácia só é calculada com feedbacks.

2. **Atualização Automática**: As estatísticas são calculadas em tempo real quando você solicita.

3. **Privacidade**: Todos os dados são armazenados localmente no SQLite.

4. **Melhoria Contínua**: Quanto mais feedback você coletar, melhor será a análise da acurácia do modelo.

## 🔄 Próximos Passos

- [ ] Adicionar gráficos de evolução da acurácia ao longo do tempo
- [ ] Exportar estatísticas para CSV/JSON
- [ ] Dashboard administrativo com mais detalhes
- [ ] Notificações quando acurácia cair abaixo de um threshold

---

**Data:** Dezembro 2024
**Versão:** 1.0 - Sistema de Feedback e Acurácia
