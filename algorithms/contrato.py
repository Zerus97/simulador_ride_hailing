"""O contrato entre o simulador e os algoritmos de reposicionamento.

A cada passo, o algoritmo recebe um único objeto, decidir(obs), e lê dele só o que
precisa. Acrescentar um campo depois não quebra nenhum algoritmo (o mesmo raciocínio
das fontes de demanda).

Regra da seta: o simulador importa o contrato; os algoritmos nunca importam o
simulador.
"""
from collections import Counter
from dataclasses import dataclass

from environment.env import GradeCelulas
from environment.previsao import PrevisaoConstante, PrevisaoDemanda
from environment.tempos import TemposEntreZonas


@dataclass(frozen=True)
class RelatorioDoPasso:
    """O que aconteceu no passo `tempo`; chega ao algoritmo na observação seguinte.

    É o único canal pelo qual os algoritmos ficam sabendo de aceites e recusas. Chega
    um passo depois por causa da ordem dos eventos: quando o algoritmo decide em t, os
    motoristas ainda não responderam; ele só fica sabendo na decisão de t + 1."""
    tempo: int
    casamentos: tuple   # ((id do veículo, id do pedido, zona), ...)
    chegadas: tuple     # ((id do veículo, zona), ...): corridas e reposicionamentos que terminaram
    aceites: tuple      # ids dos veículos que aceitaram a recomendação, de mover ou de ficar
                        # (os sem recomendação não entram nem aqui nem em recusas)
    recusas: tuple      # ids dos que recusaram (environment/comportamento.py)
    movimentos: tuple = ()  # ((id, origem, destino, tipo), ...) de todo carro que saiu da zona;
                            # tipo: "recomendado" (aceitou e se moveu), "pos_recusa" (recusou e a
                            # regra o moveu) ou "proprio" (sem recomendação, pelas preferências)


@dataclass(frozen=True)
class Observacao:
    """O que o algoritmo vê no momento da decisão, depois do matching.

    frozen=True: o algoritmo não consegue trocar um campo. Os dicionários são cópias
    montadas a cada passo, então, mesmo que o algoritmo mexa neles, o simulador não é
    afetado.

    Na hora da decisão, nenhum carro está reposicionando (os reposicionamentos
    terminam antes do matching): todo carro que não está ocioso está em corrida."""
    tempo: int               # passo da simulação (1, 2, ...)
    ociosos: tuple           # ((id, zona), ...) em ordem de id: só eles recebem recomendação
    pedidos_abertos: dict    # zona -> pedidos esperando (todas as zonas, com 0)
    chegadas: Counter        # (zona, passo de chegada) -> carros em corrida
    grade: GradeCelulas
    relatorio: RelatorioDoPasso | None = None  # o do passo anterior; None no primeiro passo
    passo_do_dia: int | None = None            # 0 a 143 (pode passar de 143 na última decisão)
    previsao: PrevisaoDemanda | PrevisaoConstante | None = None  # previsao.lam(zona, passo_do_dia)
    tempos: TemposEntreZonas | None = None     # tempos.minutos(a, b, passo_do_dia // 6), tempos.km(a, b)
    preferencias: dict | None = None           # id do ocioso -> {zona: probabilidade} (zona atual e vizinhas)
