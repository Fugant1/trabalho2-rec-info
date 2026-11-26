# Trabalho Prático 2 — Recuperação de Informação (SCC0282)

Este projeto avalia quatro métodos de recuperação de informação na coleção
NFCorpus do BEIR. Compara busca lexical com BM25, busca semântica, fusão das
duas listas e reordenação da lista híbrida. Um experimento separado mede o
efeito de erros ortográficos sobre a recuperação. Na geração aumentada por
recuperação (RAG), comparam-se respostas do mesmo modelo de linguagem com e
sem acesso aos cinco documentos recuperados para cada pergunta.

Os resultados do relatório estão em `results_final/`. A pasta `results/`
preserva a rodada de referência e a comparação do stemming em `dev`. Os
resultados oficiais foram obtidos com Snowball English no BM25 e com
`k=10` na fusão.

## Integrantes do Grupo

- **Julio Cesar Alves Araujo Fuganti** — Nº USP: `15638792`
- **Luiz Gabriel Correia dos Santos** — Nº USP: `15639682`

## Arquitetura e Modelos Escolhidos

| Componente | Implementação / Modelo | Observações |
| --- | --- | --- |
| **Dataset** | NFCorpus (BEIR) | `title` + `text` concatenados (3.633 documentos); `dev` (324 consultas) para ajuste de parâmetros, `test` (323 consultas) para avaliação final |
| **Índice Invertido** | Implementação própria do zero (`src/pre_processing/lexical_index.py`) | Armazena TF, DF, tamanhos dos documentos, `N`, `avg_doc_length`, `vocab_size`, `total_postings` e tempo de construção |
| **Busca Lexical (Baseline)** | Okapi BM25 (`src/retrieval/lexical.py`) | Opera exclusivamente sobre posting lists (sem varredura completa da coleção por consulta); `k1=1.5`, `b=0.75` (ajuste opcional no `dev`) |
| **Tolerância a Erros** | `SpellCorrector` (`src/pre_processing/typo.py`) | Dicionário de termos sem stemming, com stopwords preservadas. O ajuste em `dev` selecionou `k=2` para os k-gramas, Jaccard `≥0,20` e Damerau-Levenshtein `≤2`. |
| **Recuperação Semântica** | `sentence-transformers/all-MiniLM-L6-v2` (`src/pre_processing/semantic_index.py`, `src/retrieval/semantic.py`) | Embeddings densos normalizados L2 (384 dims), similaridade do cosseno, busca exaustiva e registro de truncamento (`max_seq_length=256`) |
| **Busca Híbrida** | Reciprocal Rank Fusion (RRF) (`src/retrieval/hybrid.py`) | Fusão dos Top-100 lexical e semântico; `k ∈ {10, 60, 100}` ajustado por `NDCG@10` no `dev` (`k=10` selecionado) |
| **Reranking** | `cross-encoder/ms-marco-MiniLM-L6-v2` (`src/retrieval/reranking.py`) | Reordena os Top-50 do ranking híbrido, preservando a ordem relativa abaixo da posição 50; registra latência e truncamento (`max_length=512`) |
| **RAG / LLM** | `Qwen/Qwen3-4B` local quantizado em 4-bit NF4 (`src/retrieval/llm.py`) | Execução 100% local via `transformers` + `bitsandbytes` (`load_in_4bit=True`, `nf4`, double quantization, `enable_thinking=False`), sem custo de API; Condição A (sem RAG) vs. Condição B (com RAG, Top-5 docs), `temperature=0.0`, `seed=42`, `max_output_tokens=1024`, idioma consistente nas duas condições |

## Ambiente e Dependências

- **Ambiente de referência:** Python `3.12.3` em Linux x86_64 sob WSL2.
- **Principais bibliotecas** (ver `requirements.txt`):
  - `torch==2.15.0`, `transformers==5.23.0`, `bitsandbytes==0.50.2`, `accelerate==1.16.0`
  - `sentence-transformers==6.3.0`
  - `beir==2.2.0`, `pytrec_eval-terrier==0.5.11`
  - `scikit-learn==1.9.1`, `numpy==2.5.4`, `pandas==3.1.1`
  - `langchain-core==1.8.1`
- **Rodadas oficiais:** WSL2 x86_64 com NVIDIA GeForce RTX 2050 de 4 GB. A recuperação usou CUDA para os modelos semântico e de reordenação. Na geração, a GPU foi usada pelo Qwen em 4 bits. Os ambientes e tempos estão nos resumos de [recuperação](results_final/experiment_summary.json) e [geração](results_final/rag/rag_execution_environment.json).

## Instalação

Os comandos usam Bash na raiz do projeto, onde estão `src/` e
`requirements.txt`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

A reprodução do RAG usa o `Qwen/Qwen3-4B` local com quantização NF4 de quatro
bits e uma GPU CUDA. Os pesos são obtidos no Hugging Face. A inferência local
dispensa chave de API e cobrança pelo uso de um serviço externo.

## Como Regenerar Dataset, Índices e Resultados

A primeira execução obtém o conjunto de dados e os pesos necessários e
prepara os índices locais. A entrega deve conter o código, a documentação
e os resultados. O conjunto de dados, os pesos, os índices em cache e o
ambiente virtual podem ser obtidos com os procedimentos a seguir.

1. **Coleção NFCorpus.** O carregador reutiliza `datasets/nfcorpus/` quando os arquivos `corpus.jsonl`, `queries.jsonl` e os três TSVs em `qrels/` estão disponíveis. Se faltarem arquivos, tenta aproveitar `datasets/nfcorpus.zip` ou baixar o ZIP do BEIR. Se o servidor do ZIP falhar, busca os textos e julgamentos nas publicações do BEIR no [Hugging Face](https://huggingface.co/datasets/BeIR/nfcorpus/tree/e28763e68d85db0fa71652ba3c4afabf8c5b3bb7) e [BeIR/nfcorpus-qrels](https://huggingface.co/datasets/BeIR/nfcorpus-qrels/tree/a451b3b26d3ae1358f259c1a3a4dd61fcea35a65). A opção `--data-dir` indica o diretório que contém `nfcorpus/`.
2. **Cache dos índices.** A execução com Snowball gera `cache/lexical_index_snowball_english.json`, `cache/semantic_embeddings.npy` e `cache/semantic_metadata.json`. O registro `runs/<run_id>/index_cache_metrics.json`, sob o diretório de resultados dos comandos abaixo, contém os tempos de construção ou de carregamento. A opção `--force-rebuild` reconstrói os índices.
3. **Modelo de linguagem.** O comando da etapa de RAG obtém a revisão dos pesos usada na rodada oficial. Os arquivos ficam no cache padrão do Hugging Face (`~/.cache/huggingface/hub`) e o modelo é quantizado durante o carregamento.

## Ordem Recomendada de Execução

Com a `.venv` ativada, os comandos abaixo gravam uma nova rodada em
`results_reproducao/` e preservam os resultados oficiais.

### 1. Executar recuperação e tolerância a erros

```bash
resultados=results_reproducao
python -m src.main --stages setup dev-tuning retrieval typo \
  --lexical-stemming snowball-english \
  --embedding-device cuda --reranker-device cuda \
  --results-dir "$resultados" --runs-dir "$resultados/runs"
```

Em um ambiente sem CUDA, usar `cpu` nas duas opções de dispositivo. Os tempos
e os resultados numéricos podem variar com o ambiente.

A execução segue a ordem dos estágios.

1. `setup`: carrega NFCorpus (`train`, `dev`, `test`) e constrói/carrega o índice invertido e os embeddings densos.
2. `dev-tuning`: ajusta `k ∈ {10, 60, 100}` do RRF e hiperparâmetros do corretor ortográfico (opcionalmente `--tune-bm25` para BM25) **exclusivamente** no conjunto `dev`.
3. `retrieval`: avalia Lexical (BM25), Semântico, Híbrido (RRF) e Híbrido + Reranking no conjunto `test` (323 consultas), valida as métricas com `pytrec_eval`, exporta os CSVs padronizados e gera as análises detalhadas por consulta (Requisitos 7 e 10).
4. `typo`: executa o experimento separado de tolerância a erros em 10 consultas elegíveis do `test` (`NDCG@10 > 0`) cobrindo inserção, remoção, substituição e troca (Requisito 4), além da avaliação opcional no sistema semântico. Pode receber casos manuais via `--typo-queries-json`.

#### Comparação do stemming no BM25

O modo padrão, `--lexical-stemming none`, usa `cache/lexical_index.json`. A opção `--lexical-stemming snowball-english` ativa o Snowball English (Porter2). Nesse modo, o índice e as consultas do BM25 passam pela mesma redução morfológica após a filtragem de stopwords. O corretor usa o vocabulário bruto do corpus, e os modelos semântico, de reranking e de RAG recebem texto sem stemming.

O modo Snowball usa `cache/lexical_index_snowball_english.json`. Os modos reaproveitam o mesmo cache semântico quando ele está disponível e válido. O índice lexical guarda o modo em `params.stemming`, e `experiment_summary.json` registra a configuração usada, incluindo a versão do Snowball quando ativo. Os comandos para reproduzir a comparação entre os dois modos em `dev` estão no [registro do ajuste lexical](results/ajuste_dev/comparacao_stemming/top100/README.md).

Os modelos de recuperação usam por padrão os identificadores da tabela de
arquitetura. No cache da rodada oficial, as revisões eram
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41` para o bi-encoder e
`233902d25c440f23af6f7d6e94d2946bac0bee0a` para o cross-encoder. Para fixar
essas versões, baixar cada modelo com `hf download <modelo> --revision <revisão>`
e passar os caminhos retornados em `--embedding-model` e `--reranker-model`.

Quando o BM25 encontra menos de 100 documentos com score positivo, completa o Top-100 com documentos de score zero. A lista de preenchimento é preparada uma vez por instância do BM25, ordenando os IDs externos e embaralhando-os com a semente fixa `2026`. A mesma ordem vale para os dois modos de stemming. O RRF e o CSV usam as posições desse ranking. O bloco `zero_score_padding` registra a semente, o SHA-256 da ordem e o número de consultas preenchidas.

### 2. Executar o experimento de RAG (sem RAG × com RAG)

O RAG lê os quatro rankings de `recuperacao/`. As dez perguntas são
sorteadas com semente 42 a partir dos IDs ordenados das 42 consultas de
`test` terminadas em ponto de interrogação. A sequência da rodada oficial
coincide com o [requisito 12 do enunciado](Trabalho_Pratico_2_RI_2026.pdf#page=5).

Na condição A (sem RAG), a entrada contém a pergunta e as instruções de
resposta. Na condição B (com RAG), acrescentam-se os cinco primeiros
documentos após a reordenação. O comando fixa a revisão dos pesos do Qwen
e gera 20 respostas em português, com temperatura zero, semente de geração
42, limite de 1.024 novos tokens e modo de raciocínio desativado. O resumo
da recuperação e o registro da geração são mantidos em arquivos distintos.

```bash
(
  set -e
  resultados=results_reproducao
  if [ -e "$resultados/rag" ]; then
    echo "O diretório $resultados/rag já existe. A reprodução requer outro diretório de resultados."
    exit 1
  fi

  modelo=$(hf download Qwen/Qwen3-4B \
    --revision 1cfa9a7208912126459214e8b04321603b3df60c \
    --include '*.json' --include '*.safetensors' --include '*.jinja' \
    --include 'merges.txt' --include 'vocab.txt' --quiet)

  mkdir -p "$resultados/runs"
  rodada_rag=$(mktemp -d "$resultados/runs/rag_local.XXXXXX")
  mkdir "$rodada_rag/recuperacao"
  cp "$resultados"/recuperacao/results_{lexical,semantic,hybrid,reranking}.csv \
    "$rodada_rag/recuperacao/"

  python -m src.main --stages rag \
    --results-dir "$rodada_rag" --runs-dir "$rodada_rag/runs" \
    --lexical-stemming snowball-english --rrf-k 10 \
    --embedding-device cpu --reranker-device cpu \
    --llm-provider local --llm-model "$modelo" --llm-language Portuguese \
    --llm-temperature 0 --llm-seed 42 --llm-max-output-tokens 1024

  cp -r "$rodada_rag/rag" "$resultados/"
  cp "$rodada_rag/experiment_summary.json" \
    "$resultados/rag/rag_execution_environment.json"
)
```

O sufixo escolhido pelo `mktemp` identifica a pasta intermediária. Ele pode
variar a cada execução. Os caminhos registrados em `files` no resumo do RAG
indicam onde os arquivos foram produzidos. As cópias usadas na avaliação ficam
em `rag/` dentro do diretório de resultados escolhido, com os mesmos nomes.
Para gerar a partir dos rankings oficiais, copiar os quatro CSVs de
`results_final/recuperacao/` para `results_reproducao/recuperacao/` antes
deste passo.

### 3. Consolidar a avaliação humana do RAG

O [guia da avaliação cega](results_final/rag/avaliacao_humana/README.md)
explica como preencher as duas cópias de `rag_blind_evaluation.csv` e consultar
o `rag_context_reference.json` da mesma rodada. Depois de concluídas as duas
avaliações, executar:

```bash
resultados=results_reproducao
avaliacoes="$resultados/rag/avaliacao_humana"
python -m src.main --stages rag-aggregate --results-dir "$resultados" \
  --blind-csv "$avaliacoes/rag_blind_luiz.csv" "$avaliacoes/rag_blind_julio.csv"
```

Usar `resultados=results_final` para consolidar a rodada oficial. A agregação
atualiza `rag/rag_experiment.csv`, registra `N/A` na rastreabilidade da
condição A e grava as médias e a concordância em
`rag/avaliacao_humana/rag_human_evaluation_summary.json`.

## Arquivos para conferência dos resultados

Os sete CSVs exigidos pelo enunciado ficam nos caminhos abaixo, relativos
à pasta `results_final/`. Os comandos de reprodução geram a mesma estrutura
em `results_reproducao/`.

| Caminho | Conteúdo |
| --- | --- |
| `recuperacao/results_lexical.csv` | Ranking do BM25, com cem documentos por consulta. |
| `recuperacao/results_semantic.csv` | Ranking semântico, com cem documentos por consulta. |
| `recuperacao/results_hybrid.csv` | Ranking após a fusão das buscas lexical e semântica. |
| `recuperacao/results_reranking.csv` | Ranking híbrido após reordenação dos primeiros cinquenta candidatos. |
| `recuperacao/metrics_per_query.csv` | Precisão, revocação, AP@10 e NDCG@10 por consulta e por método. |
| `tolerancia_erros/typo_experiment.csv` | Consultas original, com erro e corrigida, com métricas e fração da perda recuperada. |
| `rag/rag_experiment.csv` | Respostas A/B, instruções, IDs do contexto, notas e justificativas após a agregação. |

O ajuste em `dev` está em `ajuste_dev/dev_tuning_results.json`. O arquivo
`experiment_summary.json` reúne a configuração, as métricas e os tempos de
recuperação. O resumo `rag/rag_experiment_summary.json` contém as dez consultas,
os parâmetros, os tempos de geração e as métricas do contexto. O arquivo
`rag/rag_pipeline_trace.json` permite rastrear as posições dos documentos e
as evidências que chegaram a cada resposta. As métricas do contexto descrevem
os documentos enviados ao modelo. As notas de qualidade das respostas
resultam da avaliação humana.

## Testes Unitários

```bash
python -m unittest discover -s tests -v
```
