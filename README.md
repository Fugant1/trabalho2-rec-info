# Trabalho Prático 2 — Recuperação de Informação (SCC0282)

Sistema de Recuperação de Informação e Geração Aumentada por Recuperação (RAG) sobre a coleção **NFCorpus (BEIR)**, combinando busca lexical (índice invertido construído do zero + BM25), tolerância a erros ortográficos (k-gramas + Jaccard + distância de edição), recuperação semântica (bi-encoder), busca híbrida (Reciprocal Rank Fusion, o RRF), reranking neural (cross-encoder) e geração de respostas com LLM local quantizada em 4-bit (`Qwen/Qwen3-4B` via LangChain + Transformers + bitsandbytes).

## Integrantes do Grupo

- **Julio Cesar Alves Araujo Fuganti** — Nº USP: `15638792`
- **Luiz Gabriel Correia dos Santos** — Nº USP: `15639682`

## Arquitetura e Modelos Escolhidos

| Componente | Implementação / Modelo | Observações |
| --- | --- | --- |
| **Dataset** | NFCorpus (BEIR) | `title` + `text` concatenados (3.633 documentos); `dev` (324 consultas) para ajuste de parâmetros, `test` (323 consultas) para avaliação final |
| **Índice Invertido** | Implementação própria do zero (`src/pre_processing/lexical_index.py`) | Armazena TF, DF, tamanhos dos documentos, `N`, `avg_doc_length`, `vocab_size`, `total_postings` e tempo de construção (`0,43 s`) |
| **Busca Lexical (Baseline)** | Okapi BM25 (`src/retrieval/lexical.py`) | Opera exclusivamente sobre posting lists (sem varredura completa da coleção por consulta); `k1=1.5`, `b=0.75` (ajuste opcional no `dev`) |
| **Tolerância a Erros** | `SpellCorrector` (`src/pre_processing/typo.py`) | Dicionário pré-stemming/lematização (com stopwords preservadas), índice de k-gramas (`k=2`), filtro por Jaccard (`≥0.35`) e seleção por Damerau-Levenshtein (`≤2`), ajustados no `dev` |
| **Recuperação Semântica** | `sentence-transformers/all-MiniLM-L6-v2` (`src/pre_processing/semantic_index.py`, `src/retrieval/semantic.py`) | Embeddings densos normalizados L2 (384 dims), similaridade do cosseno, busca exaustiva e registro de truncamento (`max_seq_length=256`) |
| **Busca Híbrida** | Reciprocal Rank Fusion (RRF) (`src/retrieval/hybrid.py`) | Fusão dos Top-100 lexical e semântico; `k ∈ {10, 60, 100}` ajustado por `NDCG@10` no `dev` (`k=10` selecionado) |
| **Reranking** | `cross-encoder/ms-marco-MiniLM-L6-v2` (`src/retrieval/reranking.py`) | Reordena os Top-50 do ranking híbrido, preservando a ordem relativa abaixo da posição 50; registra latência e truncamento (`max_length=512`) |
| **RAG / LLM** | `Qwen/Qwen3-4B` local quantizado em 4-bit NF4 (`src/retrieval/llm.py`) | Execução 100% local via `transformers` + `bitsandbytes` (`load_in_4bit=True`, `nf4`, double quantization, `enable_thinking=False`), sem custo de API; Condição A (sem RAG) vs. Condição B (com RAG, Top-5 docs), `temperature=0.0`, `seed=42`, `max_output_tokens=1024`, idioma consistente nas duas condições |

## Ambiente e Dependências

- **Linguagem:** Python `3.12.3` (compatível com Python 3.12+)
- **Principais bibliotecas** (ver `requirements.txt`):
  - `torch==2.15.0`, `transformers==5.23.0`, `bitsandbytes==0.50.2`, `accelerate==1.16.0`
  - `sentence-transformers==6.3.0`
  - `beir==2.2.0`, `pytrec_eval-terrier==0.5.11`
  - `scikit-learn==1.9.1`, `numpy==2.5.4`, `pandas==3.1.1`
  - `langchain-core==1.8.1`
- **Hardware empregado nos experimentos:**
  - **CPU:** Intel Core i7-13650HX (x86_64, 20 threads)
  - **GPU:** NVIDIA GeForce RTX 3050 6GB Laptop GPU (CUDA 13.2)
  - **Memória:** execução otimizada com quantização 4-bit NF4 + `low_cpu_mem_usage=True` (~2,8 GB VRAM, compatível com máquinas de 16 GB de RAM)
  - **SO:** Linux x86_64 (detalhes registrados automaticamente em `results/experiment_summary.json`)

## Instalação

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

A etapa de RAG utiliza por padrão o modelo aberto **`Qwen/Qwen3-4B`** executado **localmente em 4-bit** (baixado automaticamente do Hugging Face Hub na primeira execução), portanto **não requer chave de API externa nem serviço pago**.

_(Opcionalmente, caso deseje usar Google Gemini via API, basta informar `--llm-provider gemini --llm-model gemini-3.5-flash-lite` e exportar `GOOGLE_API_KEY`)_

## Como Regenerar Dataset, Índices e Resultados

Arquivos grandes (`datasets/`, `cache/`, `.venv/`, pesos do Hugging Face) e arquivos de credenciais (`.env`) não são incluídos na entrega final; todos os dados e índices são regenerados automaticamente pelos scripts:

1. **Dataset NFCorpus:** o `DataLoader` reutiliza `datasets/nfcorpus/` quando os arquivos `corpus.jsonl`, `queries.jsonl` e os três TSVs em `qrels/` estão disponíveis. Se faltarem arquivos, tenta aproveitar `datasets/nfcorpus.zip` ou baixar o ZIP do BEIR. Se o servidor do ZIP falhar, busca os textos e julgamentos nas publicações do BEIR no [Hugging Face](https://huggingface.co/datasets/BeIR/nfcorpus/tree/e28763e68d85db0fa71652ba3c4afabf8c5b3bb7) e [BeIR/nfcorpus-qrels](https://huggingface.co/datasets/BeIR/nfcorpus-qrels/tree/a451b3b26d3ae1358f259c1a3a4dd61fcea35a65). Use `--data-dir` para indicar outra pasta que contenha `nfcorpus/`.
2. **Cache de índices (lexical e semântico):** gerados automaticamente pelo `PreProcessor` em `cache/lexical_index.json`, `cache/semantic_embeddings.npy` e `cache/semantic_metadata.json` (com custos de construção/carregamento registrados em `runs/<run_id>/index_cache_metrics.json`). Use `--force-rebuild` para forçar a reconstrução.
3. **Modelo LLM local (`Qwen/Qwen3-4B`):** baixado automaticamente para o cache padrão do Hugging Face (`~/.cache/huggingface/hub`) e quantizado em 4-bit NF4 em tempo de carregamento.

## Ordem Recomendada de Execução

### 1. Executar todas as etapas de recuperação e tolerância a erros

```bash
python -m src.main --stages setup dev-tuning retrieval typo
```

Isso executa em ordem:
1. `setup`: carrega NFCorpus (`train`, `dev`, `test`) e constrói/carrega o índice invertido e os embeddings densos.
2. `dev-tuning`: ajusta `k ∈ {10, 60, 100}` do RRF e hiperparâmetros do corretor ortográfico (opcionalmente `--tune-bm25` para BM25) **exclusivamente** no conjunto `dev`.
3. `retrieval`: avalia Lexical (BM25), Semântico, Híbrido (RRF) e Híbrido + Reranking no conjunto `test` (323 consultas), valida as métricas com `pytrec_eval`, exporta os CSVs padronizados e gera as análises detalhadas por consulta (Requisitos 7 e 10).
4. `typo`: executa o experimento separado de tolerância a erros em 10 consultas elegíveis do `test` (`NDCG@10 > 0`) cobrindo inserção, remoção, substituição e troca (Requisito 4), além da avaliação opcional no sistema semântico. Pode receber casos manuais via `--typo-queries-json`.

#### Stemming opcional no BM25

O modo padrão, `--lexical-stemming none`, usa `cache/lexical_index.json`. Para ativar Snowball English (Porter2), use `--lexical-stemming snowball-english`. Nesse modo, o índice e as consultas do BM25 passam pela mesma redução morfológica após a filtragem de stopwords. O corretor usa o vocabulário bruto do corpus, e os modelos semântico, de reranking e de RAG recebem texto sem stemming.

```bash
python -m src.main --stages setup dev-tuning retrieval typo \
  --lexical-stemming snowball-english --results-dir results_stemming
```

O modo Snowball usa `cache/lexical_index_snowball_english.json`. Os modos reaproveitam o mesmo cache semântico quando ele está disponível e válido. O índice lexical guarda o modo em `params.stemming`, e `experiment_summary.json` registra a configuração usada, incluindo a versão do Snowball quando ativo. Ao comparar execuções, use um `--results-dir` para cada modo. Para executar `rag` sobre esses rankings, repita o `--lexical-stemming` e o `--results-dir` usados na recuperação.

Quando o BM25 encontra menos de 100 documentos com score positivo, completa o Top-100 com documentos de score zero. A lista de preenchimento é preparada uma vez por instância do BM25, ordenando os IDs externos e embaralhando-os com a semente fixa `2026`. A mesma ordem vale para os dois modos de stemming. O RRF e o CSV usam as posições desse ranking. O bloco `zero_score_padding` registra a semente, o SHA-256 da ordem e o número de consultas preenchidas.

### 2. Executar o experimento de RAG (sem RAG × com RAG)

```bash
python -m src.main --stages rag --llm-model Qwen/Qwen3-4B --llm-language Portuguese
```

Reutiliza os rankings salvos em `results/recuperacao/results_*.csv`, seleciona as 10 consultas fixas com `seed=42` (`PLAIN-68`, `PLAIN-2450`, `PLAIN-153`, `PLAIN-2850`, `PLAIN-2750`, `PLAIN-270`, `PLAIN-2470`, `PLAIN-2430`, `PLAIN-238`, `PLAIN-3342`), calcula `Precision@5`, `Recall@5` e `Hit@5`, gera as respostas das Condições A e B no mesmo idioma com `Qwen/Qwen3-4B` (4-bit NF4) e produz:
- `results/rag/rag_experiment.csv`
- `results/rag/avaliacao_humana/rag_blind_evaluation.csv` (planilha embaralhada para avaliação humana cega)
- `results/rag/avaliacao_humana/rag_blind_key.json` (chave de desmascaramento A/B)
- `results/rag/avaliacao_humana/rag_context_reference.json` (documentos Top-5 completos para verificação de Fidelidade)
- `results/rag/rag_pipeline_trace.json` (rastreamento completo nas 6 etapas do pipeline para análise de falhas — Requisito 14)

_(Alternativamente, execute todo o pipeline de uma só vez com `python -m src.main --stages all`.)_

### 3. Consolidar avaliação humana do RAG (cega, 1 ou 2 avaliadores)

Após preencher as notas (`0`, `1`, `2`) e justificativas em cópias de `results/rag/avaliacao_humana/rag_blind_evaluation.csv` (consultando `results/rag/avaliacao_humana/rag_context_reference.json` para verificar Fidelidade e Rastreabilidade):

```bash
python -m src.main --stages rag-aggregate \
    --blind-csv results/rag/avaliacao_humana/rag_blind_avaliador1.csv results/rag/avaliacao_humana/rag_blind_avaliador2.csv
```

Isso desmascara as condições A e B, preenche automaticamente `N/A` na Rastreabilidade da Condição A, atualiza `results/rag/rag_experiment.csv` e calcula as médias por condição e a concordância entre avaliadores (concordância exata e Kappa de Cohen) em `results/rag/avaliacao_humana/rag_human_evaluation_summary.json`.

## Arquivos Padronizados de Resultados (`results/`)

O [guia de resultados](results/README.md) indica o que abrir para conferir cada experimento e descreve os arquivos auxiliares.

- `results_lexical.csv`: Top-100 por consulta do sistema lexical BM25 (`query_id, doc_id, rank, score, relevance`)
- `results_semantic.csv`: Top-100 por consulta do sistema semântico (`query_id, doc_id, rank, score, relevance`)
- `results_hybrid.csv`: Top-100 por consulta da busca híbrida RRF (`query_id, doc_id, rank, score, relevance`)
- `results_reranking.csv`: Top-100 por consulta após reranking com cross-encoder (`query_id, doc_id, rank, score, relevance`)
- `metrics_per_query.csv`: `P10`, `R10`, `AP10` e `NDCG10` por consulta para cada modelo
- `typo_experiment.csv`: resultados do experimento de tolerância a erros (original, erro, corrigida e taxa de recuperação)
- `rag_experiment.csv`: experimento A/B do RAG com prompts, respostas, `context_doc_ids`, notas humanas e justificativas

## Testes Unitários

```bash
python -m unittest discover -s tests -v
```
