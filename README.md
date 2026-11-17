# Voice of Customer Intelligence com RAG

## 1. Problema de negócio

O projeto utiliza avaliações textuais de clientes do **Brazilian E-Commerce Public Dataset by Olist** para permitir que gestores investiguem reclamações, elogios e padrões da experiência de compra por meio de perguntas em linguagem natural.

A solução implementa um pipeline Retrieval-Augmented Generation (RAG). As respostas são produzidas com base em comentários recuperados da base e acompanhadas das respectivas evidências.

## 2. Arquitetura

```text
CSV de avaliações da Olist
          |
          v
Preparação e limpeza (pandas)
          |
          v
Embeddings multilíngues (Sentence Transformers)
          |
          v
Banco vetorial persistente (ChromaDB)
          |
Pergunta do usuário
          |
          v
Embedding da pergunta
          |
          v
Busca semântica (Top-K)
          |
          v
Filtro por distância + construção do contexto
          |
          v
LLM local (Ollama / Llama 3.2)
          |
          v
Resposta + avaliações usadas como evidência
```

## 3. Tecnologias

- Python: implementação do pipeline.
- pandas: leitura, seleção e preparação dos dados.
- Sentence Transformers (`paraphrase-multilingual-MiniLM-L12-v2`): geração de embeddings para textos em português e outros idiomas.
- ChromaDB: armazenamento persistente e recuperação por similaridade vetorial.
- Ollama com `llama3.2:3b`: geração local das respostas.
- python-dotenv: leitura de configurações do arquivo `.env`.

## 4. Preparação da base de conhecimento

O arquivo utilizado é `olist_order_reviews_dataset.csv`, disponível no [Kaggle — Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce/data).

Coloque o CSV em `data/raw/olist_order_reviews_dataset.csv`.

A preparação seleciona os campos `review_id`, `order_id`, `review_score`, `review_comment_message` e `review_creation_date`. Comentários ausentes são substituídos por texto vazio e removidos da indexação, pois não oferecem conteúdo para busca semântica. Também são removidos registros sem identificador ou nota válida e duplicidades de `review_id`.

Cada comentário é tratado como um documento independente. Essa escolha é adequada porque avaliações costumam ser textos curtos e a separação por avaliação preserva a rastreabilidade da evidência. A nota, a data e o identificador do pedido são armazenados como metadados.

## 5. Embeddings, armazenamento e recuperação

O modelo multilíngue converte cada comentário em um vetor numérico. Esses vetores são armazenados no ChromaDB com distância cosseno.

Quando o usuário faz uma pergunta, o sistema gera seu embedding e consulta os cinco comentários mais semelhantes (`TOP_K=5`). Em seguida, aplica um limite de distância (`MAX_DISTANCE=0.75`) para evitar utilizar evidências muito distantes semanticamente. Os valores são parâmetros iniciais e podem ser ajustados após testes com perguntas e resultados conhecidos.

## 6. Geração fundamentada e limitações

Os comentários recuperados, acompanhados de seus metadados, formam o contexto enviado ao modelo local. O prompt orienta o modelo a responder somente com base nesse contexto, não inventar informações e explicitar quando as evidências forem insuficientes.

Se nenhuma avaliação atingir o limite de similaridade, o sistema informa que não encontrou evidências suficientes. As evidências também são exibidas ao usuário, permitindo conferir os comentários que contribuíram para a resposta.

A recuperação semântica não garante, por si só, que todos os comentários relevantes sejam encontrados. O limite de distância deve ser validado com perguntas representativas. Além disso, as respostas refletem apenas os comentários textuais indexados, não necessariamente toda a experiência dos clientes ou todos os dados disponíveis na Olist.

## 7. Instalação e execução

### Pré-requisitos

- Python 3.10 ou superior.
- Ollama instalado.
- Conexão com a internet na primeira execução para baixar o modelo de embeddings e o modelo do Ollama.

### Passo 1 — Criar e ativar o ambiente virtual

Windows:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python -m venv venv
source venv/bin/activate
```

### Passo 2 — Instalar dependências

```bash
pip install -r requirements.txt
```

### Passo 3 — Preparar os modelos do Ollama

Com o Ollama instalado e em execução:

```bash
ollama pull llama3.2:3b
```

O modelo pode ser alterado no arquivo `.env`.

### Passo 4 — Disponibilizar o dataset

Baixe o dataset no Kaggle e coloque `olist_order_reviews_dataset.csv` dentro de `data/raw/`.

### Passo 5 — Criar a base vetorial

Na raiz do projeto:

```bash
python src/ingest.py
```

### Passo 6 — Consultar a base

```bash
python src/app.py
```

Digite perguntas como:

- Quais problemas os clientes relatam sobre a entrega?
- O que os clientes elogiam nas avaliações?
- Quais reclamações aparecem em avaliações com nota baixa?
- Existem comentários sobre produtos com problemas de qualidade?

Para encerrar, digite `sair`.

## 8. Exemplos de perguntas e comportamento esperado

| Pergunta | Comportamento esperado |
|---|---|
| Quais problemas são relatados sobre a entrega? | Recuperar comentários semanticamente relacionados a entrega e resumir somente os problemas presentes nas evidências. |
| O que os clientes elogiam? | Recuperar avaliações com relatos positivos e identificar os aspectos mencionados. |
| Quais reclamações aparecem em avaliações com nota baixa? | Recuperar comentários relacionados à pergunta; a nota aparece como metadado para conferência da evidência. |
| Qual é a política de troca da empresa? | Caso não existam comentários relevantes, informar que a base não contém evidências suficientes para responder. |

Os exemplos descrevem perguntas para testar após a ingestão. Não são apresentados como resultados executados, pois dependem do dataset e dos modelos instalados no ambiente de execução.

## 9. Estrutura do repositório

```text
voice_of_customer_rag/
├── data/
│   ├── raw/
│   └── processed/
├── chroma_db/
├── src/
│   ├── app.py
│   ├── config.py
│   ├── ingest.py
│   ├── prepare_data.py
│   └── rag.py
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
└── roteiro_video.md
```

## 10. Decisões técnicas

- **Um comentário por documento:** mantém a unidade da avaliação e facilita apresentar a evidência original.
- **Modelo multilíngue:** permite gerar embeddings para comentários em português.
- **ChromaDB persistente:** evita recalcular e reindexar os vetores a cada pergunta.
- **Busca Top-K:** limita o contexto enviado ao modelo.
- **Filtro de distância:** reduz a chance de gerar resposta com comentários pouco relacionados.
- **LLM local:** permite executar a geração sem enviar os comentários a uma API externa; o Ollama precisa estar instalado e em execução.

## 11. Reprodutibilidade

A execução é feita em duas etapas: ingestão dos comentários e consulta à base vetorial. O dataset original não é incluído no repositório, pois deve ser baixado diretamente do Kaggle. Os modelos são baixados na primeira utilização. As configurações principais estão em `.env.example`.
