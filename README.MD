# ANÁLISE DE SENTIMENTO COM PIPELINE ETL

MVP educacional que demonstra um pipeline de dados de ponta a ponta: ingestão de arquivos .bz2, ETL com SQLite (SOR, SOT, SPEC) e treinamento de um modelo de análise de sentimento. A interface interativa é construída com Streamlit.

## 📖 Documentação

A documentação completa do projeto, incluindo o Project Model Canvas (PMC), a Arquitetura de Software e o Modelo de Dados, foi desenvolvida para guiar a implementação.

## 🖥️ Como rodar o projeto no Visual Studio Code

### 1. Pré-requisitos

**Python 3.8 ou superior.**

**Os arquivos de dados train.ft.txt.bz2 e test.ft.txt.bz2 devem estar na mesma pasta do projeto.**

### 2. Abrir o projeto

Abra o VS Code, vá em File → Open Folder e escolha a pasta onde você salvou o projeto.

### 3. Criar e ativar ambiente virtual

No terminal integrado do VS Code (Ctrl+ ou View -> Terminal):

```bash
# Criar o ambiente virtual
python -m venv .venv

# Ativar no Linux/Mac
source .venv/bin/activate

# Ativar no Windows (PowerShell)
.venv\Scripts\Activate.ps1
```
⚠️ No canto inferior direito do VS Code, clique na versão do Python e selecione o interpretador que contém .venv no caminho.

### 4. Instalar dependências
Depois, com o ambiente virtual ativo, instale as dependências:
```bash
pip install -r requirements.txt
```

### 5. Rodar a aplicação Streamlit
No terminal do VS Code, execute o comando:
```bash
streamlit run app_sentimento.py
```
A aplicação abrirá automaticamente no seu navegador no endereço http://localhost:8501.

## 📂 Estrutura de pastas

**A estrutura deste projeto foi simplificada para focar no pipeline dentro de um único script.**

```
analise-sentimento-mvp/
├─ app_sentimento.py          # Script principal com a UI e toda a lógica
├─ train.ft.txt.bz2           # Dados brutos de treino (colocar aqui)
├─ test.ft.txt.bz2            # Dados brutos de teste (colocar aqui)
├─ requirements.txt           # Dependências do projeto
├─ amazon_reviews.db          # (Será criado após o 1º treino)
├─ model_sentiment/           # (Será criada após o 1º treino)
│  └─ sentiment_classifier.pickle
└─ README.md                  # Este arquivo
```

## 🚀 Deploy

Para publicar a aplicação, você pode usar o Streamlit Community Cloud. O processo geralmente envolve conectar sua conta do GitHub e apontar para o repositório do projeto.