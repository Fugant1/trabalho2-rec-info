# Comparação inicial de stemming no dev

O T04 mede se o stemming melhora os rankings do BM25. As duas rodadas usaram os mesmos 3.633 documentos, as mesmas 324 consultas e os mesmos julgamentos de relevância do NFCorpus. A escolha usa o maior NDCG@10 lexical sem arredondar. Em empate exato, fica o modo sem stemming.

| Modo | P@10 | R@10 | MAP@10 | NDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| [Sem stemming](dev-none.json) | 0,21914 | 0,10748 | 0,08013 | 0,27328 |
| [Snowball English](dev-snowball-english.json) | 0,23457 | 0,13165 | 0,09454 | 0,29776 |

O Snowball ficou à frente por 0,02448 em NDCG@10 médio. A nota subiu em 82 consultas, manteve-se em 188 e caiu em 54. Os JSONs são cópias das saídas originais do estágio e guardam as notas por consulta em `lexical_dev.per_query`.

## Reprodução

BM25 com `k1=1.5` e `b=0.75`, profundidade solicitada de 100 e métricas calculadas sobre os dez primeiros resultados. Snowball English versão 3.1.1. Ambas as execuções usaram CPU e reaproveitaram o cache semântico.

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
for modo in none snowball-english; do
  .venv/bin/python -m src.main --stages setup dev-tuning \
    --lexical-stemming "$modo" --bm25-k1 1.5 --bm25-b 0.75 --no-tune-typo \
    --embedding-device cpu \
    --results-dir "temp_notes/stemming_execucoes/dev-$modo" \
    --runs-dir "temp_notes/stemming_execucoes/runs-dev-$modo" || break
done
```

Para rodar em modo offline, deixe o dataset e o modelo semântico disponíveis localmente. O programa pode reconstruir os índices ausentes a partir desses insumos.

- Sem stemming, execução `run_20261123_162817`, índice `cache/lexical_index.json`.
- Snowball English, execução `run_20261123_162834`, índice `cache/lexical_index_snowball_english.json`.
- Base do código `ad4dcb90059b08f2b5e47ef7f9300f33cee1d78b`, com alterações locais do T04. SHA-256 de `src/experiment.py` igual a `a221c2f96c5dbfed0682c9097f94eef77745de8b02d5964cfc3f0a01174cab4c`.
- SHA-256 de `datasets/nfcorpus/qrels/dev.tsv` igual a `b1d38b5e8f78c4a5820bce2b7ec2db54911d7690dc601e76811846b211180bd8`.

Esta comparação usa a política de ranking anterior à correção da cobertura do Top-100 lexical. A escolha para a avaliação final foi confirmada na [comparação com Top-100 completo](top100/README.md).
