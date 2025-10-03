# 🤖 Dashboard de Análise de Sentimento de Avaliações da Amazon

## 📝 Descrição do Projeto

Este é um projeto acadêmico de ponta-a-ponta para análise de sentimento em avaliações de produtos. A aplicação implementa um pipeline de dados que processa texto, treina um classificador para distinguir sentimentos positivos e negativos, e utiliza Modelagem de Tópicos (LDA) para descobrir os principais motivos por trás das avaliações negativas. Os resultados são apresentados em um dashboard interativo com Streamlit, que inclui um assistente virtual (chatbot) para resumir os resultados da análise.

## ✨ Funcionalidades

* **Pipeline de ETL Robusto:** Utiliza um banco de dados SQLite para extrair, transformar e carregar os dados, garantindo um fluxo organizado e reprodutível.
* **Pré-processamento Avançado de NLP:** Aplica técnicas como remoção de stopwords e lematização para melhorar a qualidade do texto e a precisão dos modelos.
* **Classificação de Sentimento (Positivo/Negativo):** Um modelo de Regressão Logística treinado para classificar avaliações com alta acurácia.
* **Análise de Tópicos para Avaliações Negativas:** Usa o modelo LDA para identificar e agrupar os principais temas abordados nos textos com sentimento negativo.
* **Análise das Palavras Mais Influentes:** Exibe os termos que mais contribuem para uma classificação positiva ou negativa, extraídos diretamente do modelo treinado.
* **Dashboard Interativo:** Uma interface amigável construída com Streamlit que permite executar o pipeline e explorar os insights de forma dinâmica.
* **Assistente Interativo (Chatbot):** Um chatbot que responde a perguntas em linguagem natural sobre os resultados da análise, como acurácia e palavras-chave.

## 🚀 Como Executar

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/viniciuzxd/analise-causa-raiz-reviews-amazon.git](https://github.com/viniciuzxd/analise-causa-raiz-reviews-amazon.git)
    cd analise-causa-raiz-reviews-amazon
    ```

2.  **Crie um ambiente virtual (recomendado):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # No Windows: venv\Scripts\activate
    ```

3.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Baixe os dados do NLTK (primeira vez):**
    ```bash
    python setup_nltk.py
    ```

5.  **Execute a aplicação Streamlit:**
    ```bash
    streamlit run app_sentimento.py
    ```

## 🏛️ Arquitetura

O projeto segue uma arquitetura modularizada, separando as responsabilidades em:
1.  **Leitura e Carregamento de Dados:** Leitura do arquivo `.bz2`.
2.  **Pipeline de ETL:** Gerenciamento do fluxo de dados com SQLite.
3.  **Pré-processamento de NLP:** Funções dedicadas à limpeza e preparação do texto.
4.  **Pipelines de Machine Learning:** Funções para treinamento do classificador e modelagem de tópicos.
5.  **Interface do Usuário (UI):** Dashboard construído com Streamlit.