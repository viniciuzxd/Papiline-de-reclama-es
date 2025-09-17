Project Model Canvas Adaptado — Análise de Sentimento de Avaliações
Contexto
No e-commerce moderno, as avaliações de produtos são uma fonte massiva e valiosa de dados. Empresas como a Amazon coletam milhões de opiniões de clientes, que influenciam diretamente as decisões de compra de outros consumidores e a estratégia de venda das empresas. O desafio é analisar essa enorme quantidade de texto não estruturado para extrair insights úteis.

Problema a ser Respondido
Como podemos classificar automaticamente o sentimento (positivo ou negativo) de uma avaliação de produto com base apenas no seu texto? É possível construir um pipeline de dados robusto que processe, limpe e prepare esses dados para um modelo de Machine Learning?

Pergunta Norteadora
Quais palavras ou expressões são os indicadores mais fortes de uma avaliação positiva ou negativa?

É possível treinar um modelo de classificação de texto que atinja uma alta acurácia na previsão de sentimento?

Como podemos estruturar um pipeline de ETL (Extração, Transformação e Carga) usando um banco de dados SQLite para gerenciar os dados de texto desde a sua forma bruta até a versão pronta para o modelo?

Solução Proposta
Desenvolver uma aplicação educacional em Streamlit que demonstra um pipeline de ponta a ponta para análise de sentimento:

Leia os arquivos de dados brutos (train.ft.txt.bz2 e test.ft.txt.bz2) do disco local.

Implemente um pipeline de ETL com SQLite, criando as tabelas SOR (dados brutos), SOT (dados limpos) e SPEC (dados para o modelo).

Treine um modelo de Regressão Logística para classificação de sentimento, utilizando TfidfVectorizer para processar o texto.

Exiba métricas de avaliação de classificação (acurácia, precision, recall, f1-score).

Explique a importância das palavras mais influentes para o modelo.

Responda a perguntas do usuário sobre os resultados através de um chatbot regrado.

Desenho de Arquitetura
O sistema será estruturado em camadas, com o banco de dados como peça central do ETL:

Interface (app_sentimento.py): Streamlit como front-end para iniciar o pipeline e interagir com o chatbot.

Pipeline de ETL: Módulos que gerenciam a conexão com o SQLite e o fluxo de dados entre as tabelas SOR, SOT e SPEC.

Core de Machine Learning: Funções para vetorização de texto, treinamento, avaliação e explicabilidade do modelo.

Artefatos: O banco de dados (.db) e o modelo treinado (.pickle) salvos em disco.

Resultados Esperados
Um pipeline funcional que demonstre claramente as etapas de ETL.

Modelo de classificação com acurácia na faixa de 88–92% no conjunto de validação.

Relatório de métricas e uma lista das palavras mais importantes para a classificação.

Uma aplicação interativa e funcional implantada via Streamlit.

Observação Didática
O PMC é o mapa inicial do projeto, ligando contexto, problema e solução a uma implementação prática. Ele permite que o grupo alinhe objetivos antes de programar e serve como documento pedagógico para conectar gestão de projetos com ciência de dados.