
1. System of Record (SOR)
Tabela: sor_reviews

Representa os dados brutos, exatamente como são lidos e extraídos dos arquivos train.ft.txt.bz2. É a primeira camada de armazenamento, garantindo que tenhamos uma cópia fiel dos dados originais antes de qualquer modificação.

Propósito: Ingestão e arquivamento dos dados de texto brutos.

Estrutura: As colunas são criadas para armazenar o rótulo de sentimento e o texto original da avaliação. Nenhuma limpeza ou transformação é aplicada aqui.

Coluna	Tipo de Dado (SQL)	Descrição
label	INTEGER	O rótulo de sentimento: 1 (negativo) ou 2 (positivo).
text	TEXT	O texto completo e original da avaliação do produto.

Exportar para as Planilhas
2. System of Truth (SOT)
Tabela: sot_reviews

Esta camada representa a "versão única da verdade". Os dados da SOR são limpos e padronizados para garantir consistência. É a fonte confiável para a criação de tabelas de features para os modelos.

Propósito: Fornecer dados de texto limpos e consistentes para a organização.

Transformações Aplicadas:

Padronização de Texto: O texto original é processado. Uma transformação básica e essencial para NLP, como converter todo o texto para minúsculas (lowercase), é aplicada para garantir que palavras como "Good" e "good" sejam tratadas da mesma forma pelo modelo.

Coluna	Tipo de Dado (SQL)	Descrição
label	INTEGER	O rótulo de sentimento (mantido da SOR).
text_cleaned	TEXT	O texto da avaliação após a aplicação de limpezas básicas.

Exportar para as Planilhas
3. Specification (SPEC)
Tabela: spec_reviews_train

Esta é a tabela final, otimizada e pronta para ser consumida diretamente pelo pipeline de machine learning. Ela contém as features (neste caso, o texto limpo) e a variável alvo (o rótulo).

Propósito: Fornecer um conjunto de dados de treino limpo, validado e pronto para a modelagem.

Estrutura: É uma cópia direta da SOT. A separação entre SOT e SPEC é uma boa prática que permite, no futuro, a criação de outras tabelas SPEC para diferentes tipos de modelos (ex: um modelo que usa apenas substantivos) a partir da mesma base de dados limpa (SOT), sem afetar a "fonte da verdade".

Coluna	Tipo de Dado (SQL)	Descrição
label	INTEGER	Variável alvo para o modelo de classificação.
text_cleaned	TEXT	Feature de texto pronta para ser vetorizada pelo modelo.