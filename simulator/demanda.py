"""Fontes de demanda: de onde vêm os pedidos de cada passo (padrão Estratégia).

O Simulador recebe uma fonte pronta e só pergunta a ela, a cada passo, quais pedidos
chegaram: fonte.gerar(tempo, grade, rng). Toda fonte tem esse mesmo método, então o
simulador não precisa saber qual está usando, como acontece com os algoritmos.

A fonte só diz o que chegou, como dicionários; quem cria os objetos Pedido é o
simulador, porque é ele que controla os ids, o total_pedidos e o historico_pedidos.
"""
import numpy as np


class DemandaPoisson:
    """Pedidos de mentira, para os testes: Poisson(lam) pedidos por zona e passo,
    com destino sorteado entre as outras zonas."""

    def __init__(self, lam, passo_inicial=0):
        self.lam = lam
        self.passo_inicial = passo_inicial

    def passo_do_dia(self, tempo):
        """O calendário do dia simulado, a mesma interface da DemandaRegistros (não sorteia)."""
        return self.passo_inicial + (tempo - 1)

    def gerar(self, tempo, grade, rng):
        """Lista de pedidos novos do passo `tempo`, como dicionários (origem, destino, custo).

        A ordem dos sorteios é parte do contrato (a regressão depende dela): para cada
        zona, na ordem de grade.zonas(), um sorteio de Poisson e, logo em seguida, o
        sorteio do destino de cada pedido daquela zona."""
        zonas = grade.zonas()
        novos = []
        for zona in zonas:
            num_pedidos = int(rng.poisson(self.lam))
            for k in range(num_pedidos):
                outras = [z for z in zonas if z != zona]
                destino = outras[rng.integers(len(outras))]
                novos.append({"origem": zona, "destino": destino, "custo": k + 1})
        return novos


# Colunas inteiras do pedidos_{lado}.parquet, guardadas em int8/int16/int32 (etl/demanda.py).
COLUNAS_INTEIRAS = ["dia", "segundo_do_dia", "passo", "linha_o", "coluna_o", "linha_d", "coluna_d"]


class DemandaRegistros:
    """Pedidos reais de um dia: cada viagem entra com probabilidade `fracao`.

    pedidos: a tabela do etl/demanda.py (pedidos_{lado}.parquet), com células.
    dia: o dia de março do cenário (ex.: 23, dia útil; 19, sábado).
    fracao: a fração amostrada (0,15; × 0,5 ou × 1,5 na RQ2).
    passo_inicial: o passo do dia (0 a 143) do primeiro passo da simulação; com 18,
        a simulação começa às 3h (o aquecimento)."""

    def __init__(self, pedidos, dia, fracao, passo_inicial):
        self.dia = dia
        self.fracao = fracao
        self.passo_inicial = passo_inicial

        do_dia = pedidos[pedidos["dia"] == dia].copy()
        # tipos compactos: bons para guardar, não para contas (int16 * 600 "dá a volta");
        # aqui, ao carregar, tudo vira int64
        for coluna in COLUNAS_INTEIRAS:
            do_dia[coluna] = do_dia[coluna].astype(np.int64)
        # ordem estável: o i-ésimo número aleatório de um passo cai sempre na mesma viagem
        do_dia = do_dia.sort_values("segundo_do_dia", kind="stable").reset_index(drop=True)
        self.viagens = do_dia

        # passo do dia -> colunas das viagens daquele passo (o gerar só consulta)
        self.por_passo = {
            int(passo): {c: grupo[c].to_numpy() for c in
                         ["segundo_do_dia", "linha_o", "coluna_o", "linha_d", "coluna_d",
                          "duracao_min", "km"]}
            for passo, grupo in do_dia.groupby("passo", sort=True)
        }

    def passo_do_dia(self, tempo):
        """O tempo do simulador começa em 1 (o passo() soma 1 antes de chamar a demanda)."""
        return self.passo_inicial + (tempo - 1)

    def gerar(self, tempo, grade, rng):
        """Pedidos do passo: cada viagem do passo é uma moeda, rng.random() < fracao.

        O número sorteado de cada viagem não depende da fração, então, com a mesma
        semente, quem entra com 15% também entra com 22,5%: os cenários da RQ2 ficam
        encaixados. Só a demanda usa este rng, então o algoritmo não muda os pedidos."""
        viagens = self.por_passo.get(self.passo_do_dia(tempo))
        if viagens is None:
            return []
        sorteio = rng.random(len(viagens["segundo_do_dia"])) < self.fracao
        return [
            {
                "origem": grade.zona((viagens["linha_o"][i], viagens["coluna_o"][i])),
                "destino": grade.zona((viagens["linha_d"][i], viagens["coluna_d"][i])),
                "segundo_do_dia": viagens["segundo_do_dia"][i],
                "duracao_min": viagens["duracao_min"][i],
                "km": viagens["km"][i],
            }
            for i in np.flatnonzero(sorteio)
        ]
