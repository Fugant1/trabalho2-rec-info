# Resultados

Os rankings, os testes de erros e as respostas do RAG aqui são da rodada de referência do Julio. O [resumo dessa rodada](experiment_summary.json) registra a configuração, as métricas e os tempos.

| Pasta | O que você encontra |
| --- | --- |
| [`ajuste_dev/`](ajuste_dev/README.md) | Configurações avaliadas no conjunto de desenvolvimento. |
| [`recuperacao/`](recuperacao/README.md) | Rankings, métricas por consulta e comparações entre métodos. |
| [`tolerancia_erros/`](tolerancia_erros/README.md) | Consultas com erros, resultados antes e depois da correção e detalhes do corretor. |
| [`rag/`](rag/README.md) | Respostas com e sem contexto e o rastro de cada pergunta. |
| [`rag/avaliacao_humana/`](rag/avaliacao_humana/README.md) | Planilha cega, documentos de referência e agregação das notas. |

Os guias das pastas identificam os sete CSVs exigidos no [enunciado](../Trabalho_Pratico_2_RI_2026.pdf#page=6). Saídas criadas com `--results-dir` seguem a mesma estrutura. [Como executar o pipeline](../README.md#ordem-recomendada-de-execução)
