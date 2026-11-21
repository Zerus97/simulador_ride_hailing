# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Simulador de ride-hailing em Manhattan para comparar algoritmos de reposicionamento de frota (projeto de mestrado). Código, comentários e mensagens em português.

## Ambiente e comandos

O Python base não tem as dependências. Use o ambiente conda `mestrado` (o executável fica em `<miniconda>/envs/mestrado/bin/python`):

```bash
conda run -n mestrado python -m pytest -q                                  # suíte inteira (~6 s)
conda run -n mestrado python -m pytest -q tests/test_regras.py -k sorteio  # um teste
conda run -n mestrado python main.py --algoritmo hotspot --passos 50       # simulação de brinquedo (grade 5×5, Poisson)
```

Versões fixadas em `requirements.txt` (numpy 2.5.1 é exigido pela regressão: ao instalar algo, mantenha as versões fixas).

Pipeline de dados, nesta ordem (tudo em `dataset/`, fora do git; o CSV de março/2016 e o shapefile `taxi_zones/` precisam estar lá):

| Comando | Gera |
|---|---|
| `python -m etl.etl` | `viagens_manhattan.parquet` (limpeza em 9 etapas, com relatório) |
| `python -m etl.grade` | `grade_{1200,600}.csv` (células com ≥ 20 pontas nos dias de treino) |
| `python -m etl.osm --rede` | `manhattan_drive.graphml` + `.json` (rede OSM de 2016-03-01, baixada uma vez) |
| `python -m etl.osm --pontos` | `pontos_{lado}.csv` |
| `python -m etl.osm --fatores` | `fatores_hora.csv` (+ validação fora da amostra) |
| `python -m etl.osm --tempos` | `tempos_{lado}.csv`, `nao_vizinhos_{lado}.csv` |
| `python -m etl.demanda` | `pedidos_{lado}.parquet`, `lambda_{lado}.csv` |

`--explorar`, `--diagnosticos` etc. só imprimem análises. O `README.md` está desatualizado (fala em `simulador.py` e na assinatura antiga de `decidir`).

## Arquitetura

Camadas, com imports só "para baixo":

- `environment/` — camada leve (só biblioteca padrão; lê CSV com o módulo `csv`, nunca pandas): grafo de zonas (`GrafoZonas` → `GradeCelulas` → `GradeQuadrada`), `Veiculo`/`Pedido`/`Corrida`, relógio (`tempo.py`), previsão λ (`previsao.py`), tempos entre vizinhas (`tempos.py`). Todos podem importar daqui, inclusive o ETL.
- `algorithms/` — `contrato.py` (`Observacao`, `RelatorioDoPasso`) e os algoritmos. Algoritmos nunca importam o simulador.
- `simulator/` — `Simulador`, fontes de demanda (`demanda.py`) e o `gabarito.py`.
- `etl/` — pesado (pandas, geopandas, osmnx); produz as tabelas que o simulador só lê.

**Zonas.** A chave estável é a célula `(linha, coluna)`; o id da zona (0..n−1, em ordem de célula) só existe depois de montar a `GradeCelulas` (`grade.zona(celula)`, `grade.celula(zona)`). Arquivos guardam células, nunca ids. A vizinhança tem uma única fonte: `DIRECOES` em `environment/env.py` (8 vizinhas; a ordem importa, o Hotspot desempata por ela). `nao_vizinhos` remove pares nos dois sentidos. O tabuleiro real é `grade_do_csv(grade_{lado}.csv, nao_vizinhos=nao_vizinhos_do_csv(...))`.

**O passo (`Simulador.passo`)**, nesta ordem: tempo += 1 → demanda → chegadas (corridas e reposicionamentos) → expiração → matching → `algoritmo.decidir(observar())` → reposicionamento → ociosidade → monta `ultimo_relatorio`. Convenção do relógio: tudo acontece no fim da janela de 10 min. Um pedido criado em t é atendível em t e t+1 e expira em t+2. Corridas duram `passos_da_corrida(duracao_min)` = `max(1, floor(min/10 + LIMIAR_ARREDONDAMENTO))`; reposicionamento dura sempre 1 passo (estado `REPOSICIONANDO`, chega antes do matching seguinte).

**Aleatoriedade.** `SeedSequence(seed).spawn(4)`: inicial, demanda, aceite, matching — cada fonte de acaso tem seu gerador, para o algoritmo nunca alterar os pedidos. A ordem dos sorteios dentro de `DemandaPoisson.gerar` e de `matching` é parte do contrato (documentada nas docstrings); mudá-la altera a regressão.

**Fontes de demanda (padrão Estratégia).** O simulador recebe `demanda=` com `gerar(tempo, grade, rng)` e `passo_do_dia(tempo)`. `DemandaPoisson(lam)` é para os testes da grade 5×5; `DemandaRegistros(pedidos, dia, fracao, passo_inicial)` sorteia cada viagem real com uma moeda (`rng.random() < fracao`, ordem estável por `segundo_do_dia`), de modo que frações diferentes ficam encaixadas. O simulador cria os `Pedido` (ids, totais, histórico).

**Contrato com algoritmos.** `decidir(obs)` recebe uma `Observacao` congelada (cópias montadas a cada passo; campos novos entram com default `None`) e devolve `{id: zona}`: zona vizinha = mover, zona atual = ficar, id ausente = sem recomendação, `None` proibido. `validar_recomendacoes` levanta `ValueError` (não `assert`). O relatório do passo t chega em `obs.relatorio` no passo t+1.

## Convenções que importam

- **Dias.** `DIAS_TREINO`, `SABADOS_TREINO`, `DIAS_VALIDACAO` em `etl/etl.py`. Os dias de avaliação são 23/mar (útil) e 19/mar (sábado): nada desses dias entra na construção do modelo (grade, pontos, fatores, λ, limiar).
- **Nomes de colunas.** `dia` é sempre o dia do mês; `tipo_dia` é `"util"`/`"sabado"`.
- **Tipos compactos.** Os parquets guardam `int8`/`int16`; converta para `int64` ao carregar antes de fazer contas (int16 × 600 dá a volta sem aviso).
- **λ** é a tabela densa (toda célula × todo passo, zeros explícitos), soma ÷ número de dias, nunca `groupby().mean()`. Acesso direto ao dicionário, sem `.get(..., 0)`.

## Testes

- `tests/test_regressao.py` é uma fotografia das métricas (semente 0, 500 passos). Reorganização de código não pode mudar nenhum número; mudança de modelo intencional atualiza os valores e explica no commit.
- Testes com dados reais pulam sozinhos sem `dataset/` (`pytest.skip`), para rodar numa máquina sem os dados.
- `simulator/gabarito.py` é uma reimplementação independente em tempo contínuo (heapq) usada para validar o simulador; não reaproveite código do `Simulador` nele.
- Ao criar um teste, confirme que ele falha com o defeito que deveria pegar (mutação temporária na memória, sem editar o arquivo). Refatorações também são conferidas comparando histórico de pedidos e métricas de algumas simulações antes/depois.

## Commits

Só um título curto (ex.: `1.7c`, `Recorte de Manhattan`), sem corpo e sem linha `Co-Authored-By`. Commit apenas quando pedido.
