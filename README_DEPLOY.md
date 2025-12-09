# 🚀 Guia Rápido de Deploy

## Passos para Deploy

### 1. Preparar o Projeto

```bash
# Instalar dependências
pip install -r requirements.txt

# Treinar modelos (se ainda não treinou)
python train_model_improved.py

# Popular banco de dados (opcional)
python popular_banco.py

# Verificar banco de dados
python verificar_banco.py
```

### 2. Testar Localmente

```bash
# Iniciar servidor
python app.py

# Acessar em: http://localhost:5000
```

### 3. Deploy no Heroku

```bash
# Login
heroku login

# Criar app
heroku create nome-da-app

# Deploy
git init
git add .
git commit -m "Deploy inicial"
git push heroku main

# Ver logs
heroku logs --tail
```

### 4. Deploy no Railway

1. Acesse https://railway.app
2. Conecte seu repositório GitHub
3. Railway detecta automaticamente e faz deploy

### 5. Deploy no Render

1. Acesse https://render.com
2. New Web Service
3. Conecte GitHub
4. Configure:
   - Build: `pip install -r requirements.txt`
   - Start: `python app.py`

## ✅ Checklist

- [ ] Modelos treinados em `modelos_salvos/`
- [ ] Banco de dados populado
- [ ] Testes locais passando
- [ ] Arquivos de deploy criados (Procfile, runtime.txt)
- [ ] Deploy realizado
- [ ] Aplicação funcionando

## 📝 Notas

- O banco SQLite funciona localmente, mas para produção considere PostgreSQL
- Modelos podem ser grandes - verifique limites da plataforma
- A pasta `uploads/` não persiste - considere armazenamento externo

