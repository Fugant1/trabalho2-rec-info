# Comparação de stemming com Top-100 completo

**Configuração escolhida para a avaliação final:** Snowball English no BM25 e `k=10` no RRF. A seleção do stemming usa o maior NDCG@10 lexical sem arredondamento. O valor de `k` vem do ajuste do RRF com o ranking Snowball.

| Modo | P@10 | R@10 | MAP@10 | NDCG@10 | Consultas com preenchimento |
| --- | ---: | ---: | ---: | ---: | ---: |
| [`none`](dev-none.json) | 0.21944444 | 0.10750881 | 0.08014085 | 0.27352001 | 149 |
| [`snowball-english`](dev-snowball-english.json) | 0.23487654 | 0.13168337 | 0.09455244 | 0.29797128 | 134 |

A verificação direta do índice confirmou 100 documentos distintos para cada uma das 324 consultas, nos dois modos. P@10, R@10, AP@10 e NDCG@10 por consulta coincidiram entre o avaliador do projeto e o `pytrec_eval`, com tolerância absoluta de `1e-12` para arredondamento.

## Reprodução e proveniência

As rodadas usaram os mesmos julgamentos do NFCorpus, BM25 com `k1=1.5` e `b=0.75`, profundidade 100 e o cache semântico local. A lista de IDs foi ordenada e embaralhada com `random.Random(2026).shuffle` nas duas rodadas. A semente foi escolhida pelo ano do trabalho antes da execução.

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
for modo in none snowball-english; do
  .venv/bin/python -m src.main --stages setup dev-tuning \
    --lexical-stemming "$modo" --bm25-k1 1.5 --bm25-b 0.75 --no-tune-typo \
    --embedding-device cpu \
    --results-dir "temp_notes/stemming_execucoes/top100/dev-$modo" \
    --runs-dir "temp_notes/stemming_execucoes/top100/runs-dev-$modo" || break
done
```

- `none`: execução `run_20261123_165417`, `k` selecionado 10.
- `snowball-english`: execução `run_20261123_165436`, `k` selecionado 10.
- SHA-256 da ordem dos IDs nos dois modos: `efef3d874caa3e532a778c67a3f7f4d9e471655de67e14478c507bbff5dc6719`.
- SHA-256 de `qrels/dev.tsv`: `b1d38b5e8f78c4a5820bce2b7ec2db54911d7690dc601e76811846b211180bd8`.
- Base do código: `ad4dcb90059b08f2b5e47ef7f9300f33cee1d78b`, com as alterações locais do T04 e do Top-100.
- SHA-256 de `src/experiment.py`: `a7d3003f5405e70fd2316499ad99a6620575fa1fd4c381027c63824009946fbe`.
- SHA-256 de `src/retrieval/lexical.py`: `5ff483080a6d2502ac4d5a521c3eeaa6cb3f9c5b7b6479796d07d5ca74cae120`.
- SHA-256 de `src/evaluation/evaluator.py`: `f23dd17338470b6ed80639add06dbfbdcc2f50ab8573dc8d87afae9a38c71ce6`.
