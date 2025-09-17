import streamlit as st
import pandas as pd
import bz2
import os
import pickle
import time
import numpy as np
import sqlite3

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# ==============================================================================
# 0. CONFIGURAÇÕES GERAIS
# ==============================================================================
# Nomes dos arquivos de dados locais
TRAIN_FILE_PATH = "train.ft.txt.bz2"
TEST_FILE_PATH = "test.ft.txt.bz2"

# Configurações do Banco de Dados
DB_NAME = "amazon_reviews.db"
SOR_TABLE = "sor_reviews"
SOT_TABLE = "sot_reviews"
SPEC_TABLE_TRAIN = "spec_reviews_train"
SPEC_TABLE_PREDICT = "spec_reviews_predict"

# Diretório para salvar o modelo
MODEL_DIR = "model_sentiment"
if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)
MODEL_PATH = os.path.join(MODEL_DIR, "sentiment_classifier.pickle")

# ==============================================================================
# 1. MÓDULO DE BANCO DE DADOS (Adaptado para Análise de Sentimento)
# ==============================================================================

def connect_db():
    """Cria uma conexão com o banco de dados SQLite."""
    return sqlite3.connect(DB_NAME)

def create_database_and_tables():
    """Cria o DB e as tabelas SOR, SOT e SPEC se não existirem."""
    if os.path.exists(DB_NAME):
        os.remove(DB_NAME) # Remove o DB antigo para começar do zero
    
    conn = connect_db()
    cursor = conn.cursor()
    # Tabela SOR: Armazena dados brutos
    cursor.execute(f"CREATE TABLE {SOR_TABLE} (label INTEGER, text TEXT)")
    # Tabela SOT: Armazena dados limpos/transformados
    cursor.execute(f"CREATE TABLE {SOT_TABLE} (label INTEGER, text_cleaned TEXT)")
    # Tabela SPEC: Dados prontos para o modelo
    cursor.execute(f"CREATE TABLE {SPEC_TABLE_TRAIN} (label INTEGER, text_cleaned TEXT)")
    cursor.execute(f"CREATE TABLE {SPEC_TABLE_PREDICT} (text_cleaned TEXT, original_text TEXT)")
    
    conn.commit()
    conn.close()

def insert_df_to_sor(df):
    """Insere os dados de um DataFrame na tabela SOR."""
    conn = connect_db()
    df.to_sql(SOR_TABLE, conn, if_exists="replace", index=False)
    conn.close()

def run_etl_sor_to_sot():
    """Lê da SOR, aplica uma transformação simples (lowercase) e salva na SOT."""
    conn = connect_db()
    df_sor = pd.read_sql_query(f"SELECT * FROM {SOR_TABLE}", conn)
    
    # Lógica de Transformação (simples para demonstrar o conceito)
    df_sot = pd.DataFrame()
    df_sot['label'] = df_sor['label']
    df_sot['text_cleaned'] = df_sor['text'].str.lower() # Ex: deixar tudo minúsculo
    
    df_sot.to_sql(SOT_TABLE, conn, if_exists="replace", index=False)
    conn.close()

def run_etl_sot_to_spec_train():
    """Copia dados da SOT para a SPEC de treino."""
    conn = connect_db()
    df_sot = pd.read_sql_query(f"SELECT * FROM {SOT_TABLE}", conn)
    df_sot.to_sql(SPEC_TABLE_TRAIN, conn, if_exists="replace", index=False)
    conn.close()

def run_etl_for_test_data(df_test):
    """Executa um ETL simplificado para os dados de teste e salva na SPEC de previsão."""
    conn = connect_db()
    df_spec = pd.DataFrame()
    df_spec['original_text'] = df_test['text']
    df_spec['text_cleaned'] = df_test['text'].str.lower()
    df_spec.to_sql(SPEC_TABLE_PREDICT, conn, if_exists="replace", index=False)
    conn.close()

def load_data_from_db(table_name: str):
    """Carrega dados de uma tabela específica do banco de dados."""
    conn = connect_db()
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    conn.close()
    return df

# ==============================================================================
# 2. MÓDULO DE LEITURA DE ARQUIVO
# ==============================================================================
# (A função de leitura do arquivo .bz2 permanece a mesma)
@st.cache_data
def load_amazon_reviews_from_local_bz2(file_path):
    if not os.path.exists(file_path):
        st.error(f"Arquivo não encontrado: '{file_path}'. Certifique-se de que ele está na mesma pasta que o script.")
        return None
    labels, texts = [], []
    with st.spinner(f"Carregando e processando '{file_path}'... Isso pode levar alguns minutos."):
        with bz2.open(file_path, 'rt', encoding='utf-8') as file:
            for line in file:
                first_space_index = line.find(' ')
                if first_space_index != -1:
                    labels.append(int(line[:first_space_index].replace('__label__', '')))
                    texts.append(line[first_space_index + 1:].strip())
    st.success(f"Arquivo '{file_path}' carregado com sucesso!")
    return pd.DataFrame({'label': labels, 'text': texts})


# ==============================================================================
# 3. MÓDULO DO PIPELINE (Agora integrando o DB)
# ==============================================================================

def run_training_pipeline(df_train_raw, test_size_split, progress_bar, model_path_to_save):
    # Etapa 1: ETL com Banco de Dados (45%)
    progress_bar.progress(10, text="Criando banco de dados e tabelas (SOR, SOT, SPEC)...")
    create_database_and_tables()
    time.sleep(1)
    
    progress_bar.progress(20, text="Inserindo dados brutos na tabela SOR...")
    insert_df_to_sor(df_train_raw)
    time.sleep(1)

    progress_bar.progress(30, text="Executando ETL: SOR -> SOT (limpeza)...")
    run_etl_sor_to_sot()
    time.sleep(1)

    progress_bar.progress(45, text="Executando ETL: SOT -> SPEC (preparação final)...")
    run_etl_sot_to_spec_train()
    time.sleep(1)

    # Etapa 2: Treinamento do Modelo (a partir da SPEC)
    progress_bar.progress(50, text="Carregando dados da tabela SPEC para treinamento...")
    df_spec = load_data_from_db(SPEC_TABLE_TRAIN)
    
    y = df_spec["label"]
    X = df_spec['text_cleaned']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_split, random_state=42, stratify=y)
    
    progress_bar.progress(60, text="Treinando o modelo de classificação... (Isso PODE DEMORAR VÁRIOS MINUTOS)")
    nlp_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ])
    nlp_pipeline.fit(X_train, y_train)
    
    # Etapa 3: Avaliação e Salvamento
    progress_bar.progress(80, text="Avaliando modelo e extraindo palavras importantes...")
    y_pred = nlp_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Negativo (1)', 'Positivo (2)'], output_dict=True)
    metrics = {"Acurácia": f"{accuracy:.2%}", "Relatório de Classificação": report}

    vectorizer = nlp_pipeline.named_steps['tfidf']
    classifier = nlp_pipeline.named_steps['clf']
    feature_names = vectorizer.get_feature_names_out()
    coefs = classifier.coef_[0]
    importances_df = pd.DataFrame({'feature': feature_names, 'coeficiente': coefs, 'abs_coeficiente': np.abs(coefs)}).sort_values('abs_coeficiente', ascending=False).drop(columns='abs_coeficiente')

    with open(model_path_to_save, "wb") as f:
        pickle.dump(nlp_pipeline, f)
    progress_bar.progress(90, text="Modelo salvo e métricas calculadas.")
    
    return metrics, importances_df

def run_prediction_pipeline(df_test_raw, model_path_to_load):
    # Executa ETL para os dados de teste
    run_etl_for_test_data(df_test_raw)
    df_spec_predict = load_data_from_db(SPEC_TABLE_PREDICT)

    with open(model_path_to_load, 'rb') as f:
        model = pickle.load(f)
    
    # Faz predição nos dados limpos da tabela SPEC
    predictions = model.predict(df_spec_predict['text_cleaned'])
    
    result_df = pd.DataFrame()
    result_df['texto_original'] = df_spec_predict['original_text']
    result_df['sentimento_previsto'] = ["Positivo" if p == 2 else "Negativo" for p in predictions]
    
    return result_df

# ==============================================================================
# 4. MÓDULO DO CHATBOT E UI STREAMLIT
# ==============================================================================
# (Nenhuma grande mudança aqui, apenas textos e títulos)

def answer_from_metrics_adapted(question: str, task: str, metrics_dict, importances_df):
    q = (question or "").lower()
    if "importan" in q or "palavras" in q:
        top = importances_df.head(5)["feature"].tolist()
        return f"As palavras mais influentes para o modelo de {task} são: {', '.join(top)}."
    if "métric" in q or "acur" in q:
        return f"A acurácia do modelo de {task} foi de {metrics_dict.get('Acurácia', 'N/A')}."
    if "pipeline" in q:
        return "O pipeline carrega os dados para uma tabela SOR, limpa e move para SOT e SPEC, e depois treina um modelo com TF-IDF e Regressão Logística."
    return "Desculpe, não entendi. Pergunte sobre 'palavras importantes' ou 'métricas'."

st.set_page_config(page_title="Análise de Sentimento", layout="wide")

if "model_trained" not in st.session_state: st.session_state.model_trained = False
if "predictions_made" not in st.session_state: st.session_state.predictions_made = False
if "prediction_df" not in st.session_state: st.session_state.prediction_df = None
if "metrics" not in st.session_state: st.session_state.metrics = None
if "importances" not in st.session_state: st.session_state.importances = None

st.sidebar.title("Configurações do Pipeline")
with st.sidebar:
    st.header("1. Arquivos de Dados")
    st.info(f"O modelo usará os arquivos locais:\n- **Treino:** `{TRAIN_FILE_PATH}`\n- **Teste:** `{TEST_FILE_PATH}`")
    st.warning("Certifique-se de que os arquivos estão na mesma pasta que este script.")
    st.header("2. Ações do Pipeline")
    st.subheader("Treinar Novo Modelo")
    test_size = st.slider("Tamanho do conjunto de validação", 0.1, 0.4, 0.2, 0.05)
    
    if st.button("Executar Treinamento com ETL"):
        df_train_raw = load_amazon_reviews_from_local_bz2(TRAIN_FILE_PATH)
        if df_train_raw is not None:
            progress_bar = st.progress(0, text="Iniciando pipeline de ETL e treinamento...")
            metrics, importances = run_training_pipeline(df_train_raw, test_size, progress_bar, MODEL_PATH)
            st.session_state.metrics, st.session_state.importances = metrics, importances
            st.session_state.model_trained, st.session_state.predictions_made = True, False
            progress_bar.progress(100, text="Pipeline concluído!")
            time.sleep(1)
            progress_bar.empty()
            st.success("Modelo treinado e salvo com sucesso!")
            st.rerun()

    st.subheader("Fazer Previsões com Modelo Salvo")
    if st.button("Executar Previsão"):
        if not os.path.exists(MODEL_PATH):
            st.error("Nenhum modelo treinado foi encontrado! Treine um modelo primeiro.")
        else:
            df_test_raw = load_amazon_reviews_from_local_bz2(TEST_FILE_PATH)
            if df_test_raw is not None:
                with st.spinner("Carregando modelo e fazendo previsões..."):
                    result_df = run_prediction_pipeline(df_test_raw, MODEL_PATH)
                    st.session_state.prediction_df, st.session_state.predictions_made = result_df, True
                st.success("Previsões geradas com sucesso!")
                st.rerun()

# --- Abas Principais ---
st.title("🤖 Pipeline de Análise de Sentimento (com ETL e SQLite)")
tab_train, tab_predict, tab_chat = st.tabs(["📊 Resultados do Treino", "🚀 Previsões", "💬 Chat com o Modelo"])
with tab_train:
    st.header("Métricas do Modelo de Classificação")
    if st.session_state.metrics:
        st.metric("Acurácia", st.session_state.metrics["Acurácia"])
        st.dataframe(pd.DataFrame(st.session_state.metrics["Relatório de Classificação"]).transpose())
        st.subheader("Palavras Mais Influentes")
        st.dataframe(st.session_state.importances.head(20))
    else: st.info("Execute o treinamento para ver os resultados.")
with tab_predict:
    st.header("Resultados da Previsão")
    if st.session_state.predictions_made:
        st.dataframe(st.session_state.prediction_df)
        csv_data = st.session_state.prediction_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download em CSV", csv_data, "sentiment_predictions.csv", "text/csv")
    else: st.info("Execute uma previsão para ver os resultados.")
with tab_chat:
    st.header("Converse com o Assistente do Modelo")
    if not st.session_state.model_trained:
        st.info("Treine um modelo primeiro para poder conversar.")
    else:
        if "chat_messages" not in st.session_state: st.session_state.chat_messages = [{"role": "assistant", "content": "Olá! O modelo foi treinado. Sobre o que quer saber?"}]
        for msg in st.session_state.chat_messages: st.chat_message(msg["role"]).write(msg["content"])
        if prompt := st.chat_input():
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            response = answer_from_metrics_adapted("Análise de Sentimento", st.session_state.metrics, st.session_state.importances)
            st.session_state.chat_messages.append({"role": "assistant", "content": response})
            st.rerun()