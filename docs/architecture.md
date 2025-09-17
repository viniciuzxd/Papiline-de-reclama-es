
1. Objetivos de Arquitetura
Separação de responsabilidades: A interface do usuário (Streamlit) é isolada da lógica de negócio (core/), que agora inclui um pipeline de ETL com banco de dados.

Simplicidade (MVP): Componentes focados, baixo acoplamento e um fluxo de dados claro, demonstrando um pipeline de dados de ponta a ponta.

Reprodutibilidade: Pipelines de ETL e Machine Learning encapsulados para garantir uma execução consistente.

Evolução: A estrutura permite a fácil substituição do dataset de texto, do modelo de NLP ou a inclusão de novas etapas de ETL.

Governança: Estrutura organizada para facilitar a manutenção e a documentação do projeto.

2. Visão em Camadas
A arquitetura foi adaptada para incluir o banco de dados como peça central do ETL.

Usuário (navegador)
    ↓
[ app.py ]  → Streamlit UI (Camada de Apresentação)
    ↓
[ Pipeline de Orquestração ]
    ├─ Leitura de Arquivos .bz2
    ↓
    ├─ Módulo de Banco de Dados (ETL com SQLite)
    │   ├─ Tabela SOR (Dados Brutos)
    │   ├─ Tabela SOT (Dados Limpos)
    │   └─ Tabela SPEC (Dados para o Modelo)
    ↓
    └─ Módulo de Machine Learning (Core)
        ├─ Features: Vetorização de Texto (TF-IDF)
        ├─ Modelos: Treino e Avaliação (Classificação)
        └─ Explain: Extração de Importância de Palavras
    ↓
[ Artefatos Salvos ]
    ├─ amazon_reviews.db (Banco de Dados SQLite)
    └─ model/sentiment_classifier.pickle (Modelo Treinado)
3. Componentes e Responsabilidades
app.py (Streamlit UI)
Camada de apresentação.

Fornece botões para disparar o treinamento e a previsão.

Não faz mais upload de arquivos; lê os arquivos .bz2 do diretório local.

Exibe as métricas de classificação (acurácia, etc.), as palavras mais importantes e as respostas do chatbot.

Módulo de Banco de Dados (ETL com SQLite)
SOR (System of Record): Uma tabela (sor_reviews) para receber os dados brutos (rótulo e texto) diretamente do arquivo .bz2.

SOT (System of Truth): Uma tabela (sot_reviews) que armazena os dados após uma limpeza inicial (ex: converter texto para minúsculas).

SPEC (Specific): Tabelas (spec_reviews_train, spec_reviews_predict) com os dados prontos para serem consumidos pelo modelo de Machine Learning.

Orquestra todo o fluxo de extração, transformação e carga entre as tabelas.

Módulo de Machine Learning (Core)
Features: Responsável pelo pré-processamento de texto. A principal ferramenta é o TfidfVectorizer, que transforma o texto limpo (da tabela SPEC) em vetores numéricos.

Modelos: Treina modelos de classificação de texto (como LogisticRegression). Avalia o desempenho usando métricas de classificação (acurácia, precision, recall, F1-score).

Explain: Extrai os coeficientes do modelo treinado para identificar quais palavras (features) são mais influentes para determinar se uma avaliação é positiva ou negativa.

Chatbot: Usa um conjunto de regras simples para responder perguntas sobre as métricas do modelo e as palavras mais importantes, com base nos resultados do treinamento.

Artefatos Salvos
amazon_reviews.db: O arquivo do banco de dados SQLite que contém as tabelas SOR, SOT e SPEC.

model/sentiment_classifier.pickle: O objeto do pipeline de NLP (vetorizador + classificador) treinado e salvo, pronto para ser usado em previsões futuras.

4. Fluxo de Execução
O usuário abre o aplicativo Streamlit no navegador.

O usuário clica no botão "Executar Treinamento com ETL".

A aplicação lê o arquivo train.ft.txt.bz2 do disco local.

O pipeline de ETL é iniciado:

Os dados brutos são inseridos na tabela SOR.

Os dados são lidos da SOR, transformados (limpeza de texto) e salvos na SOT.

Os dados limpos são movidos da SOT para a tabela SPEC.

O pipeline de Machine Learning é iniciado:

Lê os dados da tabela SPEC.

O TfidfVectorizer processa o texto.

O modelo LogisticRegression é treinado e avaliado.

O módulo explain extrai as palavras mais importantes.

O modelo treinado é salvo como um arquivo .pickle. O banco de dados amazon_reviews.db persiste no disco.

Os resultados (métricas, palavras importantes) são exibidos na interface do Streamlit.

5. Como desenhar no draw.io
Acesse draw.io.

Crie as seguintes caixas principais, de cima para baixo:

Usuário (Navegador)

Streamlit UI (app_sentimento.py)

Pipeline de Orquestração

Dentro da caixa "Pipeline de Orquestração", crie três caixas para representar o fluxo de ETL:

SOR (Tabela de Dados Brutos)

SOT (Tabela de Dados Limpos)

SPEC (Tabela para Modelo)

Conecte as tabelas SOR → SOT → SPEC com setas para indicar o fluxo do ETL.

Abaixo do ETL, ainda dentro do "Pipeline de Orquestração", adicione uma caixa para o Módulo de Machine Learning (Treino e Avaliação). Conecte a tabela SPEC a esta caixa.

Na base do diagrama, crie duas caixas para os artefatos:

Banco de Dados (amazon_reviews.db)

Modelo Salvo (.pickle)

Use setas para mostrar o fluxo:

Usuário → Streamlit UI (interação)

Streamlit UI → Pipeline de Orquestração (dispara o treino)

O pipeline lê de um ícone de arquivo (.bz2) e alimenta a SOR.

O Módulo de Machine Learning gera o Modelo Salvo.

O Pipeline de Orquestração gera o Banco de Dados.

Os resultados voltam para a Streamlit UI para exibição.

Salve o diagrama em formato .drawio e exporte também como .png para o seu repositório.