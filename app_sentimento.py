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
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# --- IMPORTAÇÕES DO SCIKIT-LEARN QUE ESTAVAM FALTANDO ---
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import LatentDirichletAllocation
# ---------------------------------------------------------


# --- Bloco de configuração do NLTK (executa uma vez) ---
# É seguro deixar aqui, o Streamlit gerencia o cache.
try:
    stopwords.words('english')
except LookupError:
    # Se der erro aqui, lembre-se de rodar o script setup_nltk.py primeiro
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('stopwords')
# ---------------------------------------------------------

# ==============================================================================
# 0. CONFIGURAÇÕES GERAIS (Atualizado para sua estrutura)
# ==============================================================================
# ATENÇÃO: Adicione o arquivo 'Electronics.csv' baixado do Kaggle a esta pasta.
# O código vai procurar por este nome. Os arquivos 'train.ft.txt.bz2' não serão mais usados.
DATA_FILE_PATH = "Electronics.csv"

# Um novo banco de dados será criado com este nome
DB_NAME = "electronics_reviews_v2.db"
SOR_TABLE = "sor_reviews"
SOT_TABLE = "sot_reviews"
SPEC_TABLE_TRAIN = "spec_reviews_train"

# Um novo diretório chamado 'model_complaint_v2' será criado para salvar o novo modelo,
# para não confundir com sua pasta 'model_sentiment' antiga.
MODEL_DIR = "model_complaint_v2"
if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)
MODEL_PATH = os.path.join(MODEL_DIR, "complaint_classifier_v2.pickle")

# Configurações para Modelagem de Tópicos
N_TOPICS = 5

# ==============================================================================
# 1. MÓDULO DE PRÉ-PROCESSAMENTO DE TEXTO
# ==============================================================================
def preprocess_text(text):
    """Aplica uma limpeza completa no texto: minúsculas, remove pontuação/números,
    remove stopwords e aplica lematização."""
    if not isinstance(text, str):
        return ""
    
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]
    
    return " ".join(lemmatized_tokens)

# ==============================================================================
# 2. MÓDULO DE BANCO DE DADOS
# ==============================================================================
def connect_db():
    return sqlite3.connect(DB_NAME)

def create_database_and_tables():
    if os.path.exists(DB_NAME):
        os.remove(DB_NAME)
    
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute(f"""
        CREATE TABLE {SOR_TABLE} (
            asin TEXT, 
            overall INTEGER, 
            reviewText TEXT,
            reviewTime TEXT
        )
    """)
    cursor.execute(f"""
        CREATE TABLE {SOT_TABLE} (
            asin TEXT, 
            is_complaint INTEGER, 
            text_processed TEXT,
            review_date DATE
        )
    """)
    cursor.execute(f"""
        CREATE TABLE {SPEC_TABLE_TRAIN} (
            asin TEXT, 
            is_complaint INTEGER, 
            text_processed TEXT,
            review_date DATE
        )
    """)
    conn.commit()
    conn.close()

def run_etl_pipeline(df_raw, progress_bar):
    conn = connect_db()
    df_raw.to_sql(SOR_TABLE, conn, if_exists="replace", index=False)
    
    df_sor = pd.read_sql_query(f"SELECT * FROM {SOR_TABLE}", conn)
    
    df_sot = pd.DataFrame()
    df_sot['asin'] = df_sor['asin']
    
    progress_bar.progress(30, text="ETL: Aplicando pré-processamento avançado no texto...")
    df_sot['text_processed'] = df_sor['reviewText'].apply(preprocess_text)
    
    df_sot['is_complaint'] = df_sor['overall'].apply(lambda x: 1 if x <= 2 else 0)
    df_sot['review_date'] = pd.to_datetime(df_sor['reviewTime'])
    
    df_sot.to_sql(SOT_TABLE, conn, if_exists="replace", index=False)
    
    df_spec = pd.read_sql_query(f"SELECT * FROM {SOT_TABLE}", conn)
    df_spec.to_sql(SPEC_TABLE_TRAIN, conn, if_exists="replace", index=False)
    conn.close()

def load_data_from_db(table_name: str):
    conn = connect_db()
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    conn.close()
    return df

# ==============================================================================
# 3. MÓDULO DE LEITURA DE ARQUIVO
# ==============================================================================
@st.cache_data
def load_electronics_reviews_from_csv(file_path, sample_size=None):
    if not os.path.exists(file_path):
        st.error(f"Arquivo não encontrado: '{file_path}'. Certifique-se de que 'Electronics.csv' está na pasta.")
        return None
    
    with st.spinner(f"Carregando e processando '{file_path}'..."):
        try:
            # ATENÇÃO: Verifique se 'reviewTime' é o nome correto da coluna no seu CSV!
            df = pd.read_csv(file_path, usecols=['asin', 'overall', 'reviewText', 'reviewTime'])
            df.dropna(subset=['reviewText', 'asin', 'reviewTime'], inplace=True)
            df = df[df['overall'] != 3]
            
            if sample_size and sample_size < len(df):
                df = df.sample(n=sample_size, random_state=42)

            st.success(f"Arquivo '{file_path}' carregado! {len(df)} linhas serão analisadas.")
            return df
        except Exception as e:
            st.error(f"Erro ao ler o arquivo CSV: {e}. Verifique o nome das colunas.")
            return None

# ==============================================================================
# 4. MÓDULO DO PIPELINE DE MACHINE LEARNING
# ==============================================================================
def run_complaint_classifier_pipeline(df_spec, test_size_split):
    y = df_spec["is_complaint"]
    X = df_spec['text_processed']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_split, random_state=42, stratify=y)
    
    nlp_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ])
    nlp_pipeline.fit(X_train, y_train)
    
    y_pred = nlp_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Não Reclamação', 'Reclamação'], output_dict=True)
    metrics = {"Acurácia": f"{accuracy:.2%}", "Relatório de Classificação": report}

    with open(MODEL_PATH, "wb") as f: pickle.dump(nlp_pipeline, f)
    return metrics

def run_topic_modeling_pipeline(complaint_texts, n_topics):
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=1000, stop_words='english')
    X = vectorizer.fit_transform(complaint_texts)
    
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)
    
    feature_names = vectorizer.get_feature_names_out()
    topics = {}
    topic_names = {
        0: "Bateria e Energia", 1: "Qualidade/Defeito do Produto", 2: "Conectividade e Software",
        3: "Componentes (tela, cabo)", 4: "Uso Geral e Expectativa"
    }
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-10 - 1:-1]]
        topic_name = topic_names.get(topic_idx, f"Motivo {topic_idx + 1}")
        topics[topic_name] = ", ".join(top_words)
        
    topic_distribution = lda.transform(X)
    dominant_topic_idx = np.argmax(topic_distribution, axis=1)
    dominant_topic_name = [topic_names.get(i, f"Motivo {i+1}") for i in dominant_topic_idx]
    
    return topics, dominant_topic_name

# ==============================================================================
# 5. MÓDULO UI STREAMLIT
# ==============================================================================
st.set_page_config(page_title="Análise de Reclamações", layout="wide")
st.sidebar.title("Configurações da Análise")

with st.sidebar:
    st.header("1. Dados de Entrada")
    st.info(f"O modelo usará o arquivo: `{DATA_FILE_PATH}`")
    sample_size = st.slider("Nº de avaliações para analisar (amostra)", 5000, 100000, 20000, 5000)
    
    st.header("2. Ações do Pipeline")
    test_size = st.slider("Tamanho do conjunto de validação (%)", 0.1, 0.4, 0.2, 0.05)
    
    if st.button("🚀 Executar Análise Completa"):
        st.session_state.clear()
        df_raw = load_electronics_reviews_from_csv(DATA_FILE_PATH, sample_size)
        
        if df_raw is not None:
            progress_bar = st.progress(0, text="Iniciando pipeline...")
            
            progress_bar.progress(10, text="ETL: Criando banco de dados...")
            create_database_and_tables()
            time.sleep(1)
            
            run_etl_pipeline(df_raw, progress_bar)
            
            progress_bar.progress(40, text="Carregando dados da tabela SPEC...")
            df_spec = load_data_from_db(SPEC_TABLE_TRAIN)
            df_spec['review_date'] = pd.to_datetime(df_spec['review_date'])
            st.session_state.df_spec = df_spec
            
            progress_bar.progress(50, text="Treinando modelo de classificação...")
            metrics = run_complaint_classifier_pipeline(df_spec, test_size)
            st.session_state.classifier_metrics = metrics
            
            progress_bar.progress(75, text="Analisando os motivos das reclamações (LDA)...")
            df_complaints = df_spec[df_spec['is_complaint'] == 1].copy()
            if not df_complaints.empty:
                topics, dominant_topic = run_topic_modeling_pipeline(df_complaints['text_processed'], N_TOPICS)
                st.session_state.topics = topics
                df_complaints['topic_name'] = dominant_topic
                st.session_state.df_complaints_with_topics = df_complaints
            
            progress_bar.progress(100, text="Análise concluída!")
            time.sleep(2)
            progress_bar.empty()
            st.success("Pipeline executado com sucesso!")
            st.rerun()

st.title("🤖 Dashboard de Análise de Reclamações de Produtos Eletrônicos")

if 'df_spec' not in st.session_state:
    st.info("👈 Configure e execute a análise na barra lateral para ver os resultados.")
else:
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Desempenho do Classificador", 
        "🏆 Produtos com Mais Reclamações", 
        "🔍 Análise de Motivos (O Porquê)",
        "📈 Análise Temporal (NOVO!)"
    ])

    with tab1:
        st.header("Métricas do Modelo de Classificação de Reclamações")
        if 'classifier_metrics' in st.session_state:
            metrics = st.session_state.classifier_metrics
            st.metric("Acurácia", metrics["Acurácia"])
            st.dataframe(pd.DataFrame(metrics["Relatório de Classificação"]).transpose())
        else: st.warning("Métricas não disponíveis.")

    with tab2:
        st.header("Quais produtos recebem mais reclamações?")
        if 'df_complaints_with_topics' in st.session_state:
            df_complaints = st.session_state.df_complaints_with_topics
            complaint_counts = df_complaints['asin'].value_counts().reset_index()
            complaint_counts.columns = ['asin', 'count']
            
            top_n = st.slider("Selecione o número de produtos para exibir", 5, 50, 10, key='top_n_slider')
            top_products = complaint_counts.head(top_n)

            fig = px.bar(top_products, x='asin', y='count', title=f'Top {top_n} Produtos com Mais Reclamações', labels={'asin': 'ID do Produto (ASIN)', 'count': 'Número de Reclamações'})
            fig.update_xaxes(type='category')
            st.plotly_chart(fig, use_container_width=True)
        else: st.info("Não há dados de reclamações para exibir.")

    with tab3:
        st.header("Por que os clientes estão reclamando?")
        if 'topics' in st.session_state and 'df_complaints_with_topics' in st.session_state:
            topics = st.session_state.topics
            df_complaints = st.session_state.df_complaints_with_topics

            st.subheader("Principais Motivos de Reclamação (Geral)")
            st.table(pd.DataFrame.from_dict(topics, orient='index', columns=['Palavras-chave']))
            
            st.subheader("Análise por Produto Específico")
            top_product_list = df_complaints['asin'].value_counts().head(20).index.tolist()
            selected_product = st.selectbox("Selecione um produto para analisar:", top_product_list)

            if selected_product:
                product_complaints = df_complaints[df_complaints['asin'] == selected_product]
                topic_counts = product_complaints['topic_name'].value_counts().reset_index()
                fig = px.pie(topic_counts, names='topic_name', values='count', title=f"Motivos de Reclamação para o Produto: {selected_product}", hole=0.3)
                st.plotly_chart(fig, use_container_width=True)
        else: st.info("Não há dados de tópicos para exibir.")

    with tab4:
        st.header("Como as reclamações evoluem ao longo do tempo?")
        df_complaints_time = st.session_state.get('df_complaints_with_topics')

        if df_complaints_time is not None and not df_complaints_time.empty:
            df_complaints_time['review_date'] = pd.to_datetime(df_complaints_time['review_date'])
            
            complaints_over_time = df_complaints_time.set_index('review_date').resample('M').size().reset_index(name='count')
            complaints_over_time['review_date'] = complaints_over_time['review_date'].dt.strftime('%Y-%m')

            fig = px.line(complaints_over_time, x='review_date', y='count', title='Total de Reclamações por Mês', markers=True,
                          labels={'review_date': 'Mês', 'count': 'Número de Reclamações'})
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Análise Temporal por Produto Específico")
            top_product_list_time = df_complaints_time['asin'].value_counts().head(20).index.tolist()
            selected_product_time = st.selectbox("Selecione um produto para ver sua evolução temporal:", top_product_list_time)

            if selected_product_time:
                product_df = df_complaints_time[df_complaints_time['asin'] == selected_product_time]
                product_over_time = product_df.set_index('review_date').resample('M').size().reset_index(name='count')
                product_over_time['review_date'] = product_over_time['review_date'].dt.strftime('%Y-%m')
                
                fig_product = px.line(product_over_time, x='review_date', y='count', 
                                      title=f'Evolução das Reclamações para o Produto: {selected_product_time}', markers=True,
                                      labels={'review_date': 'Mês', 'count': 'Número de Reclamações'})
                st.plotly_chart(fig_product, use_container_width=True)

        else:
            st.info("Não há dados de reclamações para exibir a análise temporal.")