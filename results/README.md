# Resultados (rodada preliminar de referência)

> **Atenção:** os resultados definitivos reportados no relatório e os sete CSVs
> padronizados para avaliação estão em **[`../results_final/`](../results_final/)**.
> Os arquivos desta pasta (`results/`) pertencem a uma rodada preliminar sem
> *stemming* e anterior à correção de preenchimento do Top-100 lexical (em que
> 149 das 323 consultas de `test` ficavam com menos de 100 documentos em
> `results_lexical.csv`), além de preservar a
> [comparação de stemming no dev](ajuste_dev/comparacao_stemming/top100/README.md)
> que embasou a configuração oficial adotada em `results_final/`.

| Pasta | O que você encontra |
| --- | --- |
| [`ajuste_dev/`](ajuste_dev/README.md) | Configurações avaliadas no conjunto de desenvolvimento. |
| [`recuperacao/`](recuperacao/README.md) | Rankings, métricas por consulta e comparações entre métodos. |
| [`tolerancia_erros/`](tolerancia_erros/README.md) | Consultas com erros, resultados antes e depois da correção e detalhes do corretor. |
| [`rag/`](rag/README.md) | Respostas com e sem contexto e o rastro de cada pergunta. |
| [`rag/avaliacao_humana/`](rag/avaliacao_humana/README.md) | Planilha cega, documentos de referência e agregação das notas. |

Os guias das pastas identificam os sete CSVs exigidos no [enunciado](../Trabalho_Pratico_2_RI_2026.pdf#page=6). Saídas criadas com `--results-dir` seguem a mesma estrutura. [Como executar o pipeline](../README.md#ordem-recomendada-de-execução)
