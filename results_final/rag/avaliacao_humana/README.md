# Avaliação humana das respostas do RAG

A rodada oficial contém duas respostas para cada uma das dez perguntas. Cada
integrante avalia as 20 respostas em uma cópia da planilha, de forma independente.
As condições são reveladas depois que as duas planilhas estiverem completas.
A escala segue o requisito 13 do enunciado (p. 6).

## 1. Preparação

| Arquivo | Uso |
| --- | --- |
| [rag_blind_evaluation.csv](rag_blind_evaluation.csv) | Planilha original, com as respostas e os campos de avaliação em branco. |
| [rag_context_reference.json](rag_context_reference.json) | Título, texto e ID dos cinco documentos de referência de cada pergunta. Para localizar os documentos, buscar pelo `query_id` da linha na planilha e consultar `top5_context_docs`. |
| [rag_blind_key.json](rag_blind_key.json) | Chave que associa cada `blind_id` à condição sem RAG (A) ou com RAG (B). Seu uso fica reservado à agregação. |

Os registros individuais estão em [rag_blind_luiz.csv](rag_blind_luiz.csv) e
[rag_blind_julio.csv](rag_blind_julio.csv). Para repetir a avaliação, cada
integrante deve começar por uma cópia da planilha original e consultar apenas
essa cópia e os documentos de referência até concluir seu julgamento. A chave
A/B fica reservada para a agregação.

Ao abrir a planilha em um editor de tabelas, importar e exportar como CSV em
UTF-8 sem BOM, com vírgula como separador e aspas duplas como delimitador de
texto. Preservar os nomes das colunas, os identificadores e o conteúdo das
respostas. Preencher apenas as quatro colunas de nota e as quatro colunas de
justificativa.

## 2. Avaliação de cada resposta

Aplicar a mesma rubrica a todas as respostas. A avaliação deve ser feita
sem identificar a condição e sem levar em conta se o modelo recebeu ou não
documentos durante a geração. Para julgar a fidelidade, consultar o mesmo
conjunto de referência nas duas respostas de cada pergunta. Para julgar a
rastreabilidade, examinar as fontes citadas na própria resposta.

1. Ler a pergunta em `query` e a resposta em `answer`.
2. Localizar o `query_id` no JSON e consultar os cinco documentos de referência.
3. Atribuir uma nota inteira de 0 a 2 por critério e registrar uma justificativa
   breve na coluna correspondente, como `justificativa_fidelidade`. Em
   fidelidade e rastreabilidade, indicar a afirmação relevante e, quando
   houver evidência, o `doc_id` e o trecho que sustenta a nota.

| Critério | 0 | 1 | 2 |
| --- | --- | --- | --- |
| Fidelidade | Afirmações importantes carecem de apoio nos documentos ou os contradizem. | Parte das afirmações é sustentada. | As afirmações principais são sustentadas. |
| Completude | Os aspectos principais da pergunta ficam sem resposta. | Parte dos aspectos é abordada. | Os aspectos que podem ser respondidos são cobertos adequadamente. |
| Utilidade | A resposta deixa de atender à necessidade expressa na pergunta. | A resposta é parcialmente útil. | A resposta atende diretamente à necessidade. |
| Rastreabilidade | As fontes das afirmações ficam indeterminadas. | Apenas parte das afirmações pode ser associada às fontes citadas. | As afirmações principais podem ser associadas aos `doc_id` citados. |

Na etapa cega, a rastreabilidade recebe nota em todas as linhas com base
nas atribuições visíveis na resposta. Quando não houver fonte identificável,
a nota provisória é 0, com a justificativa correspondente. Na consolidação,
o programa identifica a condição A, registra sua rastreabilidade como `N/A`
e a exclui da média e da concordância desse critério.

## 3. Conferência e agregação

Antes de rodar o comando, conferir se cada cópia contém os mesmos 20
`blind_id` da planilha original, uma vez cada, e se todas as notas e
justificativas foram preenchidas. O agregador aceita campos vazios e pode
produzir médias parciais, por isso a conferência deve ocorrer antes da execução.

Executar na raiz do repositório, com as duas planilhas completas:

```bash
avaliacoes=results_final/rag/avaliacao_humana

.venv/bin/python -m src.main --stages rag-aggregate \
  --results-dir results_final \
  --blind-csv "$avaliacoes/rag_blind_luiz.csv" "$avaliacoes/rag_blind_julio.csv"
```

O comando usa a chave da própria rodada para associar cada resposta à
condição A ou B. As duas planilhas individuais ficam como registro das notas
originais. A ordem dos arquivos no comando identifica Luiz como
avaliador 1 e Julio como avaliador 2.

- [rag_experiment.csv](../rag_experiment.csv) é atualizado com a média das
  duas notas por resposta e critério e com as justificativas dos integrantes.
  Uma média pode ser fracionária mesmo que as notas individuais sejam inteiras.
- `rag_human_evaluation_summary.json` registra as médias por condição em
  `condition_means` e a concordância exata e o kappa de Cohen por critério
  em `inter_annotator_agreement`. Com as duas planilhas completas, esperam-se
  20 pares de notas para fidelidade, completude e utilidade, e 10 para
  rastreabilidade, que considera apenas a condição B.
