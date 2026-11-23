# Ajuste no conjunto de desenvolvimento

A etapa `dev-tuning` gera [dev_tuning_results.json](dev_tuning_results.json). Ele registra os valores de `k` testados para combinar os rankings lexical e semântico com RRF, o resultado de cada configuração e o valor escolhido. Quando habilitado, também registra o ajuste dos parâmetros do corretor ortográfico e do BM25.

Nas novas execuções, a seção `lexical_dev` registra o modo de stemming, os parâmetros do BM25, as métricas em `global` e as notas de cada consulta em `per_query`. Para comparar os modos, abra `global → NDCG@10` no arquivo de cada rodada.

A [comparação com Top-100 completo](comparacao_stemming/top100/README.md) registra a escolha do modo de stemming e do `k` do RRF no `dev`. A [comparação inicial](comparacao_stemming/README.md) documenta os resultados anteriores ao preenchimento de rankings. As métricas da avaliação final ficam no [guia de recuperação](../recuperacao/README.md).
