# ==============================================================================
# 1. IMPORTAÇÕES (Bloco completo e corrigido)
# ==============================================================================
import streamlit as st
import pandas as pd
import os
import pickle
import time
import numpy as np
import sqlite3
import plotly.express as px
import re
import nltk
import bz2
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import LatentDirichletAllocation

# --- Bloco de configuração do NLTK ---
try:
    stopwords.words('english')
except LookupError:
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('stopwords')

# ==============================================================================
# 2. CONFIGURAÇÕES GERAIS (Adaptado para os arquivos .bz2)
# ==============================================================================
# Usando o arquivo de treino que você baixou
TRAIN_FILE_PATH = "train.ft.txt.bz2"

DB_NAME = "sentiment_reviews.db"
SOR_TABLE = "sor_reviews"
SOT_TABLE = "sot_reviews"
SPEC_TABLE_TRAIN = "spec_reviews_train"

MODEL_DIR = "model_sentiment_v2"
if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)
MODEL_PATH = os.path.join(MODEL_DIR, "sentiment_classifier_v2.pickle")

N_TOPICS = 5 # Número de tópicos para descobrir nas avaliações negativas

# ==============================================================================
# 3. MÓDULOS DE LEITURA E BANCO DE DADOS (Adaptados)
# ==============================================================================

@st.cache_data
def load_reviews_from_bz2(file_path, sample_size=None):
    """Função para ler os arquivos .bz2 com formato __label__X."""
    if not os.path.exists(file_path):
        st.error(f"Arquivo não encontrado: '{file_path}'. Certifique-se de que ele está na pasta do projeto.")
        return None
    
    labels, texts = [], []
    with st.spinner(f"Carregando e processando '{file_path}'..."):
        with bz2.open(file_path, 'rt', encoding='utf-8') as file:
            for line in file:
                try:
                    # O formato é: __label__X texto da avaliação
                    space_index = line.find(' ')
                    label = int(line[9:space_index]) # Pega o número depois de __label__
                    text = line[space_index + 1:].strip()
                    labels.append(label)
                    texts.append(text)
                except Exception:
                    continue # Ignora linhas mal formatadas
    
    df = pd.DataFrame({'label': labels, 'text': texts})

    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)

    st.success(f"Arquivo '{file_path}' carregado! {len(df)} linhas serão analisadas.")
    return df

def preprocess_text(text):
    """Função de limpeza de texto com lematização."""
    if not isinstance(text, str): return ""
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()  # <-- ALTERAÇÃO FEITA AQUI
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]
    return " ".join(lemmatized_tokens)

def run_etl_pipeline(df_raw):
    """Executa o pipeline de ETL: Raw -> SOR -> SOT -> SPEC."""
    if os.path.exists(DB_NAME): os.remove(DB_NAME)
    conn = sqlite3.connect(DB_NAME)

    # SOR
    df_raw.to_sql(SOR_TABLE, conn, if_exists="replace", index=False)

    # SOT
    df_sor = pd.read_sql_query(f"SELECT * FROM {SOR_TABLE}", conn)
    df_sot = df_sor.copy()
    df_sot['text_processed'] = df_sor['text'].apply(preprocess_text)
    df_sot.to_sql(SOT_TABLE, conn, if_exists="replace", index=False)

    # SPEC
    df_spec = pd.read_sql_query(f"SELECT label, text_processed FROM {SOT_TABLE}", conn)
    df_spec.to_sql(SPEC_TABLE_TRAIN, conn, if_exists="replace", index=False)
    
    conn.close()

# ==============================================================================
# 4. MÓDULOS DE MACHINE LEARNING
# ==============================================================================

def run_classification_pipeline(df_spec, test_size_split):
    """Treina o classificador de sentimento (Positivo vs. Negativo)."""
    df_spec['label'] = df_spec['label'].map({1: 'Negativo', 2: 'Positivo'})
    
    y = df_spec["label"]
    X = df_spec['text_processed']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_split, random_state=42, stratify=y)
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    metrics = {
        "Acurácia": f"{accuracy_score(y_test, y_pred):.2%}",
        "Relatório de Classificação": classification_report(y_test, y_pred, output_dict=True)
    }

    # Extrair importância das palavras
    vectorizer = pipeline.named_steps['tfidf']
    classifier = pipeline.named_steps['clf']
    feature_names = vectorizer.get_feature_names_out()
    coefs = classifier.coef_[0]
    importances_df = pd.DataFrame({
        'palavra': feature_names, 
        'importancia': coefs
    }).sort_values('importancia', ascending=False)
    
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)
        
    return metrics, importances_df

def run_topic_modeling_pipeline(df_spec, n_topics):
    """Executa a modelagem de tópicos nas avaliações negativas."""
    negative_texts = df_spec[df_spec['label'] == 1]['text_processed']
    if negative_texts.empty or len(negative_texts) < n_topics:
        return None  
    
    vectorizer = TfidfVectorizer(max_df=0.9, min_df=5, max_features=1000, stop_words='english')
    X = vectorizer.fit_transform(negative_texts)
    
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)
    
    feature_names = vectorizer.get_feature_names_out()
    topics = {}
    for topic_idx, topic_words in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic_words.argsort()[:-8:-1]]
        topics[f"Motivo Negativo {topic_idx + 1}"] = ", ".join(top_words)
        
    return topics

# ==============================================================================
# FUNÇÃO DO CHATBOT (OPCIONAL)
# ==============================================================================
def chatbot_answer(question, metrics, importances_df, topics):
    """Gera respostas para o chatbot com base nos resultados da análise."""
    q = (question or "").lower()

    if "acurácia" in q or "desempenho" in q:
        return f"A acurácia do modelo de classificação foi de {metrics.get('Acurácia', 'N/A')}."

    if "positivas" in q:
        top_positive = importances_df.head(5)["palavra"].tolist()
        return f"As 5 palavras mais influentes para um sentimento POSITIVO são: {', '.join(top_positive)}."

    if "negativas" in q:
        top_negative = importances_df.tail(5).sort_values('importancia', ascending=True)["palavra"].tolist()
        return f"As 5 palavras mais influentes para um sentimento NEGATIVO são: {', '.join(top_negative)}."

    if "motivos" in q or "tópicos" in q or "causa" in q:
        if topics:
            # Pega o primeiro tópico como exemplo de resposta
            main_topic_name = next(iter(topics.keys()))
            main_topic_words = topics[main_topic_name]
            return f"Um dos principais motivos para avaliações negativas é relacionado a '{main_topic_name}', com palavras-chave como: {main_topic_words}."
        else:
            return "A análise de tópicos não encontrou resultados claros na amostra atual para eu poder resumir."
            
    return "Não entendi a pergunta. Você pode me perguntar sobre 'acurácia', 'palavras positivas', 'palavras negativas' ou 'motivos das reclamações'."

# ==============================================================================
# 5. INTERFACE DO STREAMLIT (UI)
# ==============================================================================
st.set_page_config(page_title="Análise de Sentimento", layout="wide")
st.sidebar.title("Configurações da Análise")

with st.sidebar:
    st.header("1. Dados de Entrada")
    st.info(f"O modelo usará o arquivo: `{TRAIN_FILE_PATH}`")
    sample_size = st.slider("Nº de avaliações para analisar", 10000, 500000, 50000, 10000)
    
    st.header("2. Ações do Pipeline")
    test_size = st.slider("Tamanho do conjunto de validação (%)", 0.1, 0.4, 0.2, 0.05)
    
    if st.button("🚀 Executar Análise Completa"):
        st.session_state.clear()
        df_raw = load_reviews_from_bz2(TRAIN_FILE_PATH, sample_size)
        
        if df_raw is not None:
            progress_bar = st.progress(0, text="Iniciando pipeline...")
            
            progress_bar.progress(25, text="ETL: Processando e movendo dados...")
            run_etl_pipeline(df_raw)
            
            conn = sqlite3.connect(DB_NAME)
            df_spec = pd.read_sql(f"SELECT * FROM {SPEC_TABLE_TRAIN}", conn)
            conn.close()
            
            progress_bar.progress(50, text="Treinando modelo de classificação...")
            metrics, importances = run_classification_pipeline(df_spec, test_size)
            st.session_state.metrics = metrics
            st.session_state.importances = importances
            
            progress_bar.progress(75, text="Analisando os motivos das avaliações negativas...")
            topics = run_topic_modeling_pipeline(df_spec, N_TOPICS)
            st.session_state.topics = topics

            progress_bar.progress(100, text="Análise concluída!")
            st.success("Pipeline executado com sucesso!")
            st.balloons()
            st.rerun()

st.title("🤖 Dashboard de Análise de Sentimento de Avaliações")

if 'metrics' not in st.session_state:
    st.info("👈 Configure e execute a análise na barra lateral para ver os resultados.")
else:
    tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Desempenho do Classificador", 
    "🔍 Palavras Mais Influentes",
    "📉 Motivos das Avaliações Negativas",
    "💬 Chat com o Assistente"
    ])

    with tab1:
        st.header("Métricas do Modelo de Classificação")
        st.metric("Acurácia", st.session_state.metrics["Acurácia"])
        report_df = pd.DataFrame(st.session_state.metrics["Relatório de Classificação"]).transpose()
        st.dataframe(report_df)

    with tab2:
        st.header("Palavras que Mais Influenciam o Sentimento")
        importances_df = st.session_state.importances
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Top 15 Palavras Positivas")
            st.dataframe(importances_df.head(15))
        with col2:
            st.subheader("Top 15 Palavras Negativas")
            st.dataframe(importances_df.tail(15).sort_values('importancia', ascending=True))

    with tab3:
        st.header("Análise de Tópicos nas Avaliações Negativas")
        if st.session_state.topics:
            topics_df = pd.DataFrame.from_dict(st.session_state.topics, orient='index', columns=['Principais Palavras-chave'])
            st.table(topics_df)
            st.info("Estes são os principais temas encontrados automaticamente nos textos com sentimento negativo.")
        else:
            st.warning("Não foi possível gerar os tópicos (pode haver poucas amostras negativas).")
    # Bloco de código para a nova aba do Chatbot
    with tab4:
        st.header("Converse com o Assistente de Análise")
        st.info("Faça perguntas em linguagem natural sobre os resultados do modelo.")

        # Inicializa o histórico do chat
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "Olá! O modelo foi treinado. Sobre o que você gostaria de saber?"}]

        # Mostra as mensagens do histórico
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Input do usuário
        if prompt := st.chat_input("Pergunte sobre acurácia, palavras, motivos...", key="chat_widget"):
            # Adiciona a mensagem do usuário ao histórico e à tela
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Gera e mostra a resposta do assistente
            with st.chat_message("assistant"):
                response = chatbot_answer(
                    prompt, 
                    st.session_state.metrics, 
                    st.session_state.importances, 
                    st.session_state.topics
                )
                st.markdown(response)

            # Adiciona a resposta do assistente ao histórico
            st.session_state.messages.append({"role": "assistant", "content": response})

        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "Olá! O modelo foi treinado. Sobre o que você gostaria de saber?"}]

        # Mostra as mensagens do histórico
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Input do usuário
        if prompt := st.chat_input("Pergunte sobre acurácia, palavras, motivos..."):
            # Adiciona a mensagem do usuário ao histórico e à tela
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Gera e mostra a resposta do assistente
            with st.chat_message("assistant"):
                response = chatbot_answer(
                    prompt, 
                    st.session_state.metrics, 
                    st.session_state.importances, 
                    st.session_state.topics
                )
                st.markdown(response)

            # Adiciona a resposta do assistente ao histórico
            st.session_state.messages.append({"role": "assistant", "content": response})