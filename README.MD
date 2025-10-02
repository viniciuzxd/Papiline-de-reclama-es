# 🤖 Dashboard de Análise de Reclamações de Produtos Eletrônicos

## 📝 Descrição do Projeto

Este projeto é uma aplicação web interativa construída com Streamlit que realiza uma análise de ponta a ponta das avaliações de produtos eletrônicos da Amazon. O objetivo principal é identificar quais produtos recebem mais reclamações e entender os principais motivos por trás da insatisfação dos clientes, analisando também a evolução desses sentimentos ao longo do tempo.

## ✨ Funcionalidades

* **Pipeline de ETL Robusto:** Utiliza um banco de dados SQLite para extrair, transformar e carregar os dados, garantindo um fluxo organizado desde os dados brutos até a versão pronta para modelagem.
* **Pré-processamento Avançado de NLP:** Aplica técnicas como remoção de stopwords e lematização para melhorar a qualidade do texto.
* **Classificação de Reclamações:** Um modelo de Regressão Logística treinado para classificar avaliações como "Reclamação" ou "Não Reclamação" com alta acurácia.
* **Modelagem de Tópicos (LDA):** Identifica e agrupa os principais motivos de reclamação (ex: problemas de bateria, defeitos de tela, conectividade).
* **Análise Temporal:** Visualiza a evolução do número de reclamações ao longo do tempo, tanto de forma geral quanto para produtos específicos.
* **Dashboard Interativo:** Uma interface amigável com gráficos e tabelas (criados com Plotly) que permite explorar os insights de forma dinâmica.

## 🚀 Como Executar

1.  **Clone o repositório:**
    ```bash
    git clone [URL-do-seu-repositorio]
    cd [nome-do-repositorio]
    ```

2.  **Crie um ambiente virtual (recomendado):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # No Windows: venv\Scripts\activate
    ```

3.  **Instale as dependências:**
    Certifique-se de que o arquivo `requirements.txt` está na pasta e execute:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Baixe os dados do NLTK:**
    Execute o script `setup_nltk.py` uma única vez para baixar os pacotes necessários:
    ```bash
    python setup_nltk.py
    ```

5.  **Execute a aplicação Streamlit:**
    ```bash
    streamlit run app_sentimento.py
    ```

## 🏛️ Arquitetura

O projeto segue uma arquitetura modularizada, separando as responsabilidades em:
1.  **Leitura e Carregamento de Dados:** Leitura do CSV e amostragem.
2.  **Pipeline de ETL:** Gerenciamento do fluxo de dados com SQLite (SOR, SOT, SPEC).
3.  **Pré-processamento de NLP:** Funções dedicadas à limpeza e preparação do texto.
4.  **Pipelines de Machine Learning:** Funções para treinamento do classificador e modelagem de tópicos.
5.  **Interface do Usuário (UI):** Dashboard construído com Streamlit para visualização e interatividade.