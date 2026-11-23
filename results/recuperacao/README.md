# Recuperação de documentos

A etapa `retrieval` gera os cinco CSVs exigidos para recuperação no conjunto `test`.

| Arquivo | O que mostra |
| --- | --- |
| [results_lexical.csv](results_lexical.csv) | Documentos ordenados pelo BM25. |
| [results_semantic.csv](results_semantic.csv) | Documentos ordenados pela similaridade entre os embeddings da consulta e dos documentos. |
| [results_hybrid.csv](results_hybrid.csv) | Combinação das posições dos rankings lexical e semântico por RRF. |
| [results_reranking.csv](results_reranking.csv) | Ranking híbrido após reordenar os primeiros 50 documentos com o cross-encoder. |
| [metrics_per_query.csv](metrics_per_query.csv) | P@10, R@10, AP@10 e NDCG@10 por consulta e método. A média de `AP10` por método dá o MAP@10. |

Nos rankings, filtre por `query_id` e leia na ordem de `rank`. A coluna `relevance` vem dos julgamentos da coleção e vale zero quando o par consulta-documento não tem julgamento. A escala de `score` depende do método.

### Análises por consulta

- [analysis_lexical_vs_semantic.json](analysis_lexical_vs_semantic.json) — Três maiores vantagens de cada método em NDCG@10, com documentos, posições, scores e evidências lexicais.
- [analysis_reranking.json](analysis_reranking.json) — Dois maiores ganhos, um caso de pouca mudança e o maior retrocesso, com as posições antes e depois da reordenação.

Na auditoria da rodada salva, em 21/11/2026, o BM25 tinha 100 documentos em 174 das 323 consultas.
