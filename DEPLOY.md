# Guia de Deploy - Aplicação de Análise Nutricional

Este guia explica como fazer deploy da aplicação em diferentes plataformas.

## 📋 Pré-requisitos

1. Conta na plataforma escolhida (Heroku, Railway, Render, etc.)
2. Git instalado e configurado
3. Modelos treinados na pasta `modelos_salvos/`
4. Banco de dados populado (opcional, mas recomendado)

## 🚀 Deploy no Heroku

### 1. Instalar Heroku CLI
```bash
# Windows
# Baixe de: https://devcenter.heroku.com/articles/heroku-cli

# Verificar instalação
heroku --version
```

### 2. Login no Heroku
```bash
heroku login
```

### 3. Criar aplicação
```bash
heroku create nome-da-sua-app
```

### 4. Configurar variáveis de ambiente
```bash
heroku config:set FLASK_ENV=production
heroku config:set FLASK_DEBUG=False
```

### 5. Fazer deploy
```bash
git init
git add .
git commit -m "Initial commit"
git push heroku main
```

### 6. Verificar logs
```bash
heroku logs --tail
```

## 🚂 Deploy no Railway

### 1. Conectar repositório GitHub
- Acesse https://railway.app
- Conecte seu repositório GitHub
- Railway detectará automaticamente o projeto Python

### 2. Configurar variáveis de ambiente
No painel do Railway:
- `PORT`: Será definido automaticamente
- `FLASK_ENV`: `production`
- `FLASK_DEBUG`: `False`

### 3. Deploy automático
O Railway fará deploy automaticamente a cada push no GitHub.

## 🎨 Deploy no Render

### 1. Criar novo Web Service
- Acesse https://render.com
- Clique em "New +" > "Web Service"
- Conecte seu repositório GitHub

### 2. Configurações
- **Name**: Nome da sua aplicação
- **Environment**: Python 3
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `python app.py`

### 3. Variáveis de ambiente
- `PORT`: Render define automaticamente
- `FLASK_ENV`: `production`
- `FLASK_DEBUG`: `False`

## 📦 Preparação para Deploy

### 1. Treinar modelos
```bash
python train_model_improved.py
```

### 2. Popular banco de dados (opcional)
```bash
python popular_banco.py
```

### 3. Verificar arquivos necessários
- ✅ `Procfile` - Define comando de inicialização
- ✅ `runtime.txt` - Versão do Python
- ✅ `requirements.txt` - Dependências
- ✅ `app.py` - Aplicação principal
- ✅ `database.py` - Gerenciamento do banco
- ✅ `modelos_salvos/` - Modelos treinados

## ⚠️ Notas Importantes

1. **Tamanho dos modelos**: Modelos podem ser grandes. Considere usar Git LFS ou armazenamento externo.

2. **Banco de dados**: O SQLite funciona localmente, mas para produção considere PostgreSQL ou outro banco gerenciado.

3. **Uploads**: A pasta `uploads/` não persiste em deploys. Considere usar serviços como AWS S3 ou Cloudinary.

4. **Memória**: Aplicações com modelos ML podem precisar de mais memória. Verifique os limites da plataforma.

## 🔧 Troubleshooting

### Erro: "No module named 'flask'"
```bash
pip install -r requirements.txt
```

### Erro: "Modelos não encontrados"
- Verifique se a pasta `modelos_salvos/` está no repositório
- Ou configure upload dos modelos após deploy

### Erro: "Port already in use"
- Use variável de ambiente `PORT` definida pela plataforma

## 📝 Checklist de Deploy

- [ ] Modelos treinados e salvos
- [ ] Banco de dados populado (opcional)
- [ ] `Procfile` criado
- [ ] `runtime.txt` configurado
- [ ] `requirements.txt` atualizado
- [ ] Variáveis de ambiente configuradas
- [ ] Testes locais passando
- [ ] Deploy realizado
- [ ] Aplicação funcionando em produção

## 🌐 URLs de Exemplo

Após o deploy, sua aplicação estará disponível em:
- Heroku: `https://nome-da-sua-app.herokuapp.com`
- Railway: `https://nome-da-sua-app.up.railway.app`
- Render: `https://nome-da-sua-app.onrender.com`

---

**Última atualização:** Dezembro 2024

