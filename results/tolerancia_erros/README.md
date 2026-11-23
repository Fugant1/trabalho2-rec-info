# Tolerância a erros

A etapa `typo` testa o efeito de erros de digitação na busca e quanto o corretor consegue recuperar.

| Arquivo | O que mostra |
| --- | --- |
| [typo_experiment.csv](typo_experiment.csv) | CSV exigido para esta etapa. Compara o NDCG@10 lexical de dez consultas nas versões original, com erro e corrigida. A fração recuperada é calculada quando o erro reduz a métrica. |
| [typo_experiment_details.json](typo_experiment_details.json) | Parâmetros do corretor, erros introduzidos e candidatos à correção por consulta. |
| [typo_experiment_semantic.csv](typo_experiment_semantic.csv) | Avaliação opcional das mesmas consultas na busca semântica. |
