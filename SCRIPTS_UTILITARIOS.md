# 📜 Scripts Utilitários do Projeto

## 📋 Resumo dos Scripts

Todos os scripts abaixo são **NECESSÁRIOS** e têm funções específicas importantes:

### ✅ Scripts que DEVEM ser mantidos:

#### 1. `testar_aplicacao.py`
**Função:** Verifica se a aplicação está configurada corretamente
- Testa importações de dependências
- Verifica banco de dados
- Verifica modelos
- Verifica estrutura de arquivos
**Quando usar:** Antes de iniciar a aplicação ou após instalar dependências
**Comando:** `python testar_aplicacao.py`

#### 2. `popular_banco.py`
**Função:** Popula o banco de dados com dados nutricionais do CSV
- Lê `nutrition.csv`
- Insere dados na tabela `alimentos`
- Acelera buscas nutricionais
**Quando usar:** Uma vez após criar o banco de dados ou quando atualizar o CSV
**Comando:** `python popular_banco.py`
**Mencionado no:** README.md

#### 3. `train_model_improved.py`
**Função:** Treina modelo melhorado de classificação
- Carrega dataset
- Treina Random Forest com otimização de hiperparâmetros
- Salva modelos em `modelos_salvos/`
**Quando usar:** Quando precisar treinar/retreinar modelos
**Comando:** `python train_model_improved.py`
**Tempo:** Pode demorar vários minutos/horas dependendo do dataset

#### 4. `verificar_banco.py`
**Função:** Verifica estado do banco de dados
- Lista tabelas
- Mostra estatísticas de refeições, alimentos, predições
- Verifica integridade
**Quando usar:** Para verificar se dados estão sendo salvos corretamente
**Comando:** `python verificar_banco.py`

#### 5. `verificar_acuracia.py`
**Função:** Exibe estatísticas de acurácia do modelo
- Calcula acurácia geral
- Mostra estatísticas por faixa de confiança
- Lista alimentos mais preditos
- Mostra últimas predições com feedback
**Quando usar:** Para monitorar performance do modelo
**Comando:** `python verificar_acuracia.py`

## 🚀 Ordem Recomendada de Execução

### Primeira vez (Setup inicial):
1. `python testar_aplicacao.py` - Verificar dependências
2. `python popular_banco.py` - Popular banco de dados
3. `python train_model_improved.py` - Treinar modelos (se necessário)
4. `python verificar_banco.py` - Verificar se tudo foi salvo

### Uso regular:
- `python testar_aplicacao.py` - Verificar se tudo está OK
- `python verificar_banco.py` - Ver estado do banco
- `python verificar_acuracia.py` - Ver estatísticas de acurácia

### Quando necessário:
- `python train_model_improved.py` - Retreinar modelos
- `python popular_banco.py` - Atualizar dados nutricionais

## ❌ Nenhum script deve ser excluído

Todos os scripts têm funções específicas e úteis:
- ✅ `testar_aplicacao.py` - Diagnóstico e verificação
- ✅ `popular_banco.py` - Setup inicial e atualização
- ✅ `train_model_improved.py` - Treinamento de modelos
- ✅ `verificar_banco.py` - Monitoramento do banco
- ✅ `verificar_acuracia.py` - Monitoramento de acurácia

## 📝 Notas

- Todos os scripts são independentes e podem ser executados separadamente
- Nenhum script é chamado automaticamente pela aplicação principal
- Todos são úteis para manutenção, diagnóstico e setup do projeto
- Manter todos facilita a manutenção e troubleshooting

---

**Conclusão:** Todos os scripts devem ser **MANTIDOS** - nenhum deve ser excluído.
