# Guia dos resultados

Para conferir as médias e os parâmetros da execução, abra [experiment_summary.json](experiment_summary.json). Para reproduzir as etapas, siga as [instruções de execução](../README.md#ordem-recomendada-de-execução). Os sete CSVs abaixo correspondem ao requisito 15 do [enunciado](../Trabalho_Pratico_2_RI_2026.pdf#page=6).

## CSVs obrigatórios

| Arquivo | Conteúdo | Etapa que gera |
| --- | --- | --- |
| [results_lexical.csv](results_lexical.csv) | Ranking lexical BM25 por consulta. | `retrieval` |
| [results_semantic.csv](results_semantic.csv) | Ranking por similaridade entre embeddings da consulta e dos documentos. | `retrieval` |
| [results_hybrid.csv](results_hybrid.csv) | Ranking obtido pela fusão RRF dos rankings lexical e semântico. | `retrieval` |
| [results_reranking.csv](results_reranking.csv) | Ranking híbrido após reordenação dos primeiros 50 documentos pelo cross-encoder. | `retrieval` |
| [metrics_per_query.csv](metrics_per_query.csv) | P@10, R@10, AP@10 e NDCG@10 por consulta e modelo. A média de `AP10` por modelo é o MAP@10. | `retrieval` |
| [typo_experiment.csv](typo_experiment.csv) | Dez consultas com as versões original, com erro e corrigida, seus valores de NDCG@10 lexical e a fração recuperada quando houve perda. | `typo` |
| [rag_experiment.csv](rag_experiment.csv) | Perguntas, prompts, respostas, documentos de contexto e campos de avaliação humana. A condição A usa a pergunta sem contexto e a B recebe o Top-5 reranqueado. | `rag`, atualizado por `rag-aggregate` |

Nos quatro rankings, use `query_id` para localizar a consulta e `rank` para conferir a ordem. A coluna `relevance` registra o grau dos qrels e vale zero quando o par consulta-documento não tem julgamento. O `score` tem a escala do método que produziu o ranking.

Na auditoria de 21/11/2026, o CSV lexical tinha 100 documentos em 174 das 323 consultas e os campos de avaliação humana do RAG ainda aguardavam preenchimento, com `N/A` já registrado para rastreabilidade da condição A.

## Resumos e análises

| Arquivo | Conteúdo | Etapa que gera |
| --- | --- | --- |
| [experiment_summary.json](experiment_summary.json) | Resumo consolidado do ambiente, dataset, indexação, métricas e tempos. | Atualizado ao executar as etapas do experimento |
| [dev_tuning_results.json](dev_tuning_results.json) | Configurações avaliadas no `dev`, incluindo o ajuste de `k` do RRF e do corretor ortográfico. | `dev-tuning` |
| [analysis_lexical_vs_semantic.json](analysis_lexical_vs_semantic.json) | Três maiores vantagens de cada método em NDCG@10, com documentos, posições, scores e evidências lexicais. | `retrieval` |
| [analysis_reranking.json](analysis_reranking.json) | Dois maiores ganhos, um caso de pouca mudança e o maior retrocesso, com posições antes e depois. | `retrieval` |
| [typo_experiment_details.json](typo_experiment_details.json) | Parâmetros do corretor, detalhes das alterações e candidatos por consulta. | `typo` |
| [typo_experiment_semantic.csv](typo_experiment_semantic.csv) | Extensão opcional com as mesmas dez consultas avaliadas na busca semântica. | `typo` com avaliação semântica habilitada |
| [rag_experiment_summary.json](rag_experiment_summary.json) | Configuração da LLM, latências A/B e P@5, R@5 e Hit@5 dos documentos enviados à condição B. | `rag` |
| [rag_pipeline_trace.json](rag_pipeline_trace.json) | Rastro por consulta da recuperação lexical e semântica até o contexto e a resposta. | `rag` |

## Material da avaliação humana do RAG

| Arquivo | Como usar |
| --- | --- |
| [rag_blind_evaluation.csv](rag_blind_evaluation.csv) | Planilha embaralhada. Cada avaliador preenche as notas e justificativas em sua própria cópia. |
| [rag_context_reference.json](rag_context_reference.json) | Textos completos dos cinco documentos de referência de cada pergunta, usados para avaliar fidelidade nas duas condições. |
| [rag_blind_key.json](rag_blind_key.json) | Correspondência entre `blind_id` e condição A/B, usada na agregação após a avaliação. |

Com as duas planilhas preenchidas, `rag-aggregate` atualiza `rag_experiment.csv` e gera `rag_human_evaluation_summary.json`, com as médias por condição e a concordância entre avaliadores.
