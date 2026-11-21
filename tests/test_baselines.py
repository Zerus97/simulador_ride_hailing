"""1.11: os baselines sem treino."""
from collections import Counter

import numpy as np
import pytest

from algorithms.contrato import Observacao
from algorithms.gulosa import GulosaDemanda
from algorithms.sem_movimento import SemMovimento
from algorithms.sem_recomendacao import SemRecomendacao
from algorithms.vizinho_aleatorio import VizinhoAleatorio
from environment.comportamento import AceiteFixo, Comportamento, Ficar
from environment.env import REPOSICIONANDO, GradeQuadrada
from environment.previsao import PrevisaoDemanda
from simulator.demanda import DemandaPoisson
from simulator.simulator import Simulador
from tests.test_regras import PREF_5X5

GRADE = GradeQuadrada(5)  # a zona 12 é o centro, com 8 vizinhas


def comportamento(rho=1.0, sem="ficar"):
    return Comportamento(AceiteFixo(rho), Ficar(), sem_recomendacao=sem)


def observacao(ociosos, passo_do_dia=0, previsao=None):
    return Observacao(tempo=1, ociosos=tuple(ociosos), pedidos_abertos={}, chegadas=Counter(),
                      grade=GRADE, passo_do_dia=passo_do_dia, previsao=previsao)


# --- as exigências ---

def test_sem_movimento_com_preferencias_da_erro():
    with pytest.raises(ValueError):
        Simulador(GRADE, 5, DemandaPoisson(0.4), SemMovimento(), seed=0, preferencias=PREF_5X5,
                  comportamento=comportamento(sem="preferencias"))


def test_sem_recomendacao_com_ficar_da_erro():
    with pytest.raises(ValueError):
        Simulador(GRADE, 5, DemandaPoisson(0.4), SemRecomendacao(), seed=0)  # o padrão é "ficar"


# --- sem recomendação ---

def test_sem_recomendacao_nao_depende_do_aceite():
    def rodada(rho):
        sim = Simulador(GRADE, 10, DemandaPoisson(0.4), SemRecomendacao(), seed=2, preferencias=PREF_5X5,
                        comportamento=comportamento(rho, sem="preferencias"))
        moveu = 0
        for _ in range(100):
            sim.passo()
            assert sim.ultimo_relatorio.aceites == () and sim.ultimo_relatorio.recusas == ()
            moveu += sum(v.estado == REPOSICIONANDO for v in sim.veiculos)
        return ([(p.origem, p.destino, p.criado_em) for p in sim.historico_pedidos],
                [v.zona for v in sim.veiculos], sim.metrics(), moveu)

    com_1, com_03 = rodada(1.0), rodada(0.3)
    assert com_1 == com_03
    assert com_1[3] > 0  # os carros se movem pelas preferências


# --- vizinho aleatório ---

def recomendacoes_aleatorias(n, seed):
    alg = VizinhoAleatorio()
    alg.iniciar(np.random.default_rng(seed))
    return alg.decidir(observacao((i, 12) for i in range(n)))


def test_vizinho_aleatorio_nunca_fica_e_e_uniforme():
    rec = recomendacoes_aleatorias(8000, seed=0)
    destinos = list(rec.values())
    vizinhas = GRADE.vizinhas(12)
    assert 12 not in destinos and set(destinos) == set(vizinhas)
    p = 1 / len(vizinhas)
    for z in vizinhas:
        assert abs(destinos.count(z) / 8000 - p) < 4 * np.sqrt(p * (1 - p) / 8000), z


def test_vizinho_aleatorio_reprodutivel():
    assert recomendacoes_aleatorias(50, seed=3) == recomendacoes_aleatorias(50, seed=3)
    assert recomendacoes_aleatorias(50, seed=3) != recomendacoes_aleatorias(50, seed=4)


# --- gulosa ---

def previsao(valores):
    """valores: {(zona, passo): λ}; o resto é 0."""
    lam = {(z, p): 0.0 for z in GRADE.zonas() for p in range(144)}
    lam.update(valores)
    return PrevisaoDemanda(lam)


def test_gulosa_olha_o_passo_seguinte():
    # agora (passo 50): a vizinha 7 tem mais; no passo seguinte (51), a 17
    prev = previsao({(7, 50): 9.0, (7, 51): 1.0, (17, 50): 1.0, (17, 51): 5.0})
    rec = GulosaDemanda().decidir(observacao([(0, 12), (1, 12)], passo_do_dia=50, previsao=prev))
    assert rec == {0: 17, 1: 17}  # todos os ociosos da zona vão para o mesmo lugar


def test_gulosa_passo_143_olha_o_passo_0():
    prev = previsao({(13, 0): 2.0})
    rec = GulosaDemanda().decidir(observacao([(0, 12)], passo_do_dia=143, previsao=prev))
    assert rec == {0: 13}


def test_gulosa_no_empate_fica():
    prev = previsao({(z, 51): 3.0 for z in GRADE.zonas()})
    rec = GulosaDemanda().decidir(observacao([(0, 12)], passo_do_dia=50, previsao=prev))
    assert rec == {0: 12}


def test_gulosa_sem_previsao_da_erro():
    with pytest.raises(ValueError):
        GulosaDemanda().decidir(observacao([(0, 12)]))
