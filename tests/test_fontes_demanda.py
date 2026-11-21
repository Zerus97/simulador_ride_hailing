"""Fontes de demanda do simulador (simulator/demanda.py), com pedidos de mentira."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from algorithms.sem_movimento import SemMovimento
from environment.env import GradeCelulas, grade_do_csv
from simulator.demanda import COLUNAS_INTEIRAS, DemandaRegistros
from simulator.simulator import Simulador

DATASET = Path(__file__).resolve().parent.parent / "dataset"
GRADE = GradeCelulas([(0, 0), (0, 1), (1, 0)])  # zonas 0, 1, 2


def pedidos_de_mentira(viagens):
    """viagens: (dia, passo, (linha_o, coluna_o), (linha_d, coluna_d), duracao_min).
    O segundo_do_dia é o início do passo; os inteiros vêm compactos, como no parquet."""
    tabela = pd.DataFrame([{
        "dia": dia, "segundo_do_dia": passo * 600, "passo": passo,
        "linha_o": o[0], "coluna_o": o[1], "linha_d": d[0], "coluna_d": d[1],
        "duracao_min": duracao, "km": 1.0,
    } for dia, passo, o, d, duracao in viagens])
    return tabela.astype({"dia": "int8", "segundo_do_dia": "int32", "passo": "int16",
                          "linha_o": "int16", "coluna_o": "int16", "linha_d": "int16", "coluna_d": "int16"})


def muitas_viagens(n, passo=18, dia=23):
    """n viagens no mesmo passo; a duração (0, 1, 2, ...) identifica cada uma."""
    return pedidos_de_mentira([(dia, passo, (0, 0), (0, 1), float(i)) for i in range(n)])


def gerados(fonte, tempo, seed):
    return [p["duracao_min"] for p in fonte.gerar(tempo, GRADE, np.random.default_rng(seed))]


def test_passo_certo():
    # passo do dia = passo_inicial + (tempo - 1); com 18, o tempo 1 é o passo 18 (3h00 a 3h10)
    pedidos = pedidos_de_mentira([
        (23, 17, (0, 0), (0, 1), 17.0),
        (23, 18, (0, 0), (0, 1), 18.0),
        (23, 19, (0, 0), (0, 1), 19.0),
        (22, 18, (0, 0), (0, 1), 99.0),  # outro dia: nunca entra
    ])
    fonte = DemandaRegistros(pedidos, dia=23, fracao=1.0, passo_inicial=18)
    assert gerados(fonte, tempo=1, seed=0) == [18.0]
    assert gerados(fonte, tempo=2, seed=0) == [19.0]


def test_fracao_1_e_0():
    pedidos = muitas_viagens(50)
    assert len(gerados(DemandaRegistros(pedidos, 23, 1.0, 18), 1, seed=0)) == 50
    assert gerados(DemandaRegistros(pedidos, 23, 0.0, 18), 1, seed=0) == []


def test_reprodutivel():
    fonte = DemandaRegistros(muitas_viagens(1000), 23, 0.15, 18)
    assert gerados(fonte, 1, seed=7) == gerados(fonte, 1, seed=7)
    assert gerados(fonte, 1, seed=7) != gerados(fonte, 1, seed=8)


def test_encaixe_entre_fracoes():
    # com a mesma semente, quem entra com 15% também entra com 22,5% (RQ2: × 1,5)
    pedidos = muitas_viagens(5000)
    base = set(gerados(DemandaRegistros(pedidos, 23, 0.15, 18), 1, seed=3))
    alta = set(gerados(DemandaRegistros(pedidos, 23, 0.225, 18), 1, seed=3))
    assert base < alta


def test_fracao_sorteada():
    n = 100_000
    sorteados = gerados(DemandaRegistros(muitas_viagens(n), 23, 0.15, 18), 1, seed=0)
    assert 0.147 <= len(sorteados) / n <= 0.153


def test_celulas_viram_zonas_e_inteiros_int64():
    pedidos = pedidos_de_mentira([(23, 18, (1, 0), (0, 1), 5.0)])
    fonte = DemandaRegistros(pedidos, 23, 1.0, 18)

    (pedido,) = fonte.gerar(1, GRADE, np.random.default_rng(0))
    assert pedido["origem"] == GRADE.zona((1, 0)) == 2
    assert pedido["destino"] == GRADE.zona((0, 1)) == 1
    assert all(fonte.viagens[c].dtype == np.int64 for c in COLUNAS_INTEIRAS)
    assert isinstance(pedido["segundo_do_dia"], np.int64)


# --- Com os dados reais (pula sem o dataset) ---

@pytest.mark.parametrize("dia", [23, 19])
def test_dia_real_tem_cerca_de_15_por_cento(dia):
    pedidos_arq, grade_arq = DATASET / "pedidos_1200.parquet", DATASET / "grade_1200.csv"
    if not (pedidos_arq.exists() and grade_arq.exists()):
        pytest.skip("dataset fora do repositório")
    pedidos = pd.read_parquet(pedidos_arq)
    passos = 18  # 3h às 6h
    fonte = DemandaRegistros(pedidos, dia=dia, fracao=0.15, passo_inicial=18)
    sim = Simulador(grade_do_csv(grade_arq), num_veiculos=50, demanda=fonte,
                    algoritmo=SemMovimento(), seed=0)

    sim.rodar(passos)

    n = ((pedidos["dia"] == dia) & pedidos["passo"].between(18, 18 + passos - 1)).sum()
    esperado, desvio = 0.15 * n, np.sqrt(n * 0.15 * 0.85)
    assert abs(sim.total_pedidos - esperado) < 4 * desvio, (sim.total_pedidos, esperado)
