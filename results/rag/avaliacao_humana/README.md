# Avaliação humana do RAG

| Arquivo | Como usar |
| --- | --- |
| [rag_blind_evaluation.csv](rag_blind_evaluation.csv) | Cada avaliador preenche as notas e justificativas em sua própria cópia da planilha embaralhada. |
| [rag_context_reference.json](rag_context_reference.json) | Textos completos dos cinco documentos de referência de cada pergunta, usados para julgar a fidelidade nas duas condições. |
| [rag_blind_key.json](rag_blind_key.json) | Liga cada `blind_id` à condição A ou B. Use a chave após a avaliação individual, na agregação. |

Com as planilhas preenchidas, a etapa `rag-aggregate` atualiza [rag_experiment.csv](../rag_experiment.csv) e gera `rag_human_evaluation_summary.json`, com médias por condição e concordância entre avaliadores. [Comando de agregação](../../../README.md#3-consolidar-avaliação-humana-do-rag-cega-1-ou-2-avaliadores)

Na rodada salva, as notas e justificativas estão pendentes. A rastreabilidade da condição A consta como `N/A`.
