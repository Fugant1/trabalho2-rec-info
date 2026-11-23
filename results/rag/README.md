# Geração de respostas com e sem contexto

A etapa `rag` roda duas condições para as mesmas perguntas. A condição A usa só a pergunta. A condição B usa a pergunta e os cinco primeiros documentos após o reranking.

| Arquivo | O que mostra |
| --- | --- |
| [rag_experiment.csv](rag_experiment.csv) | CSV exigido com perguntas, prompts, respostas, IDs dos documentos de contexto e campos de avaliação humana. |
| [rag_experiment_summary.json](rag_experiment_summary.json) | Configuração da LLM, tempos de geração em A e B e métricas P@5, R@5 e Hit@5 do contexto. |
| [rag_pipeline_trace.json](rag_pipeline_trace.json) | Percurso de cada pergunta da busca lexical e semântica até o contexto e a resposta. |

Para preencher ou consolidar as notas, use o [guia da avaliação humana](avaliacao_humana/README.md).
