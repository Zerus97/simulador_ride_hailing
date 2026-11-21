"""O gabarito em tempo contínuo (simulator/gabarito.py), com casos calculados à mão."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import environment.tempo as tempo
from algorithms.sem_movimento import SemMovimento
from environment.env import GradeCelulas, GradeQuadrada, grade_do_csv
from simulator.demanda import DemandaRegistros
from simulator.gabarito import taxa_gabarito
from simulator.simulator import Simulador

FIM = 24 * 3600
DATASET = Path(__file__).resolve().parent.parent / "dataset"


def pedido(segundo, origem=0, destino=0, duracao_min=5.0):
    return SimpleNamespace(segundo_do_dia=segundo, origem=origem, destino=destino, duracao_min=duracao_min)


def test_carro_ocioso_atende_na_hora_e_depois_pega_o_da_fila():
    # carro em 0; pedido às 100 s (ocupa até 400 s); outro às 200 s espera e é atendido às 400 s
    assert taxa_gabarito([pedido(100), pedido(200)], [0], FIM) == 1.0


def test_pedido_expira_se_o_carro_so_fica_livre_depois_do_prazo():
    # o 1º ocupa o carro até 100 + 25 × 60 = 1600 s; o 2º (às 200 s) tem prazo
    # (0 + 2) × 600 = 1200 s < 1600 s: expira
    assert taxa_gabarito([pedido(100, duracao_min=25), pedido(200)], [0], FIM) == 0.5


def test_prazo_e_inclusivo():
    # o carro fica livre exatamente no prazo do 2º (1200 s): ainda dá para atender
    assert taxa_gabarito([pedido(0, duracao_min=20), pedido(10)], [0], FIM) == 1.0


def test_empate_carro_livre_antes_do_pedido():
    # o carro fica livre às 400 s e um pedido chega às 400 s: o carro vem primeiro e atende
    assert taxa_gabarito([pedido(100), pedido(400, duracao_min=1)], [0], FIM) == 1.0


def test_carro_livre_vai_para_o_destino():
    # o carro leva o 1º pedido da zona 0 para a 1, e lá atende o 2º
    pedidos = [pedido(0, origem=0, destino=1), pedido(100, origem=1, destino=1)]
    assert taxa_gabarito(pedidos, [0], FIM) == 1.0


def test_atendimento_depois_do_fim_nao_conta():
    # o carro só fica livre às 400 s, depois do fim (350 s): o 2º não é atendido
    assert taxa_gabarito([pedido(100), pedido(200)], [0], fim_segundos=350) == 0.5


def test_sem_pedidos():
    assert taxa_gabarito([], [0], FIM) != taxa_gabarito([], [0], FIM)  # NaN


# --- Simulador × gabarito: o arredondamento das corridas (LIMIAR_ARREDONDAMENTO) ---

def dia_sintetico(seed=0):
    """Um dia no formato do pedidos_{lado}.parquet, numa grade 4 × 4: Poisson(60) pedidos
    por passo das 3h às 24h, segundo sorteado dentro do passo, origem e destino sorteados,
    duração lognormal(ln 9,5; 0,65), com o mesmo viés das viagens de Nova York."""
    rng = np.random.default_rng(seed)
    partes = []
    for passo in range(18, 144):
        n = rng.poisson(60)
        origem, destino = rng.integers(0, 4, (n, 2)), rng.integers(0, 4, (n, 2))
        partes.append(pd.DataFrame({
            "dia": 23, "segundo_do_dia": passo * 600 + rng.integers(0, 600, n), "passo": passo,
            "linha_o": origem[:, 0], "coluna_o": origem[:, 1],
            "linha_d": destino[:, 0], "coluna_d": destino[:, 1],
            "duracao_min": rng.lognormal(np.log(9.5), 0.65, n), "km": 1.0,
        }))
    return pd.concat(partes, ignore_index=True)


def taxas(tabela, n, limiar, monkeypatch, grade=None, fracao=1.0, seed=0):
    """Num dia de 3h às 24h (dia 23 da tabela), com o limiar dado:
    (taxa do simulador, taxa do gabarito, total de pedidos)."""
    monkeypatch.setattr(tempo, "LIMIAR_ARREDONDAMENTO", limiar)
    grade = GradeQuadrada(4) if grade is None else grade
    sim = Simulador(grade, n, DemandaRegistros(tabela, 23, fracao, 18), SemMovimento(), seed=seed)
    zonas_iniciais = [v.zona for v in sim.veiculos]
    sim.rodar(126)
    gabarito = taxa_gabarito(sim.historico_pedidos, zonas_iniciais, (18 + 126) * 600)
    return sim.metrics()["taxa_atendimento"], gabarito, sim.total_pedidos


def comparar(tabela, n, monkeypatch, **kw):
    """Roda c = 0 e c = 0,5 e confere que as duas rodadas sortearam a mesma amostra
    (a demanda tem gerador próprio): senão o gabarito compararia amostras diferentes.
    Devolve (gabarito, c = 0 − gabarito, c = 0,5 − gabarito)."""
    baixo, gabarito, total_baixo = taxas(tabela, n, 0.0, monkeypatch, **kw)
    meio, gabarito_meio, total_meio = taxas(tabela, n, 0.5, monkeypatch, **kw)
    assert total_baixo == total_meio and gabarito == gabarito_meio
    return gabarito, baixo - gabarito, meio - gabarito


def test_arredondamento_para_baixo_bate_com_o_gabarito_no_dia_sintetico(monkeypatch):
    gabarito, dif_baixo, dif_meio = comparar(dia_sintetico(), 40, monkeypatch)
    assert dif_meio <= -0.03, (gabarito, dif_meio)   # c = 0,5: o mínimo infla as corridas
    assert abs(dif_baixo) < 0.01, (gabarito, dif_baixo)


# --- Com o dia 23 real (pulam sem o dataset) ---

@pytest.fixture(scope="module")
def pedidos_reais():
    arquivo = DATASET / "pedidos_1200.parquet"
    if not arquivo.exists():
        pytest.skip("dataset fora do repositório")
    return pd.read_parquet(arquivo)


def manhattan_em_uma_zona(pedidos):
    """Todas as células viram a (0, 0): qualquer carro atende qualquer pedido."""
    uma = pedidos.copy()
    uma[["linha_o", "coluna_o", "linha_d", "coluna_d"]] = 0
    return uma


# Zona única: o extremo SEM gargalo espacial, como se o reposicionamento fosse perfeito
# e instantâneo; horários e durações continuam reais. Varredura de 21/nov (dia 23, 0,15):
#   frota   gabarito   c=0 − gab   c=0,5 − gab
#    200      41,2%     +0,75        −4,49
#    300      59,6%     +1,60        −6,21
#    400      77,1%     +2,67        −7,61
#    500      92,7%     +3,13        −7,84
#    600     100,0%     +0,02        −2,74
# Limites: c = 0,5 abaixo por pelo menos metade do menor erro (6,21 / 2 ≈ 3,1); c = 0 a
# menos de 1,5 vez o maior desvio (1,5 × 2,67 ≈ 4,0). c = 0 fica ACIMA do gabarito porque,
# no dia 23, o arredondamento deixa as corridas 0,24 min mais curtas que as reais (nos dias
# de treino, onde o limiar foi calibrado, só 0,06); com a frota no limite da capacidade,
# isso vira atendimento a mais.
@pytest.mark.parametrize("frota", [300, 400])
def test_uma_zona_c_meio_erra_e_c_zero_acompanha_o_gabarito(pedidos_reais, frota, monkeypatch):
    gabarito, dif_baixo, dif_meio = comparar(
        manhattan_em_uma_zona(pedidos_reais), frota, monkeypatch,
        grade=GradeCelulas([(0, 0)]), fracao=0.15)
    assert 0.5 <= gabarito <= 0.8
    assert dif_meio <= -0.031, (gabarito, dif_meio)
    assert abs(dif_baixo) < 0.040, (gabarito, dif_baixo)


# 58 zonas com SemMovimento: o extremo DO gargalo espacial. Os carros ficam presos em zonas
# sem demanda (no pico, todos os ociosos estão em zonas sem pedido esperando), então a
# duração das corridas pesa pouco e c = 0,5 quase não erra; só c = 0 é exigido.
# Varredura de 21/nov: c = 0 − gabarito = +0,37 (1.200 carros) e +0,20 (2.000).
@pytest.mark.parametrize("frota", [1200, 2000])
def test_58_zonas_c_zero_acompanha_o_gabarito(pedidos_reais, frota, monkeypatch):
    grade_arq = DATASET / "grade_1200.csv"
    if not grade_arq.exists():
        pytest.skip("dataset fora do repositório")
    gabarito, dif_baixo, _ = comparar(pedidos_reais, frota, monkeypatch,
                                      grade=grade_do_csv(grade_arq), fracao=0.15)
    assert 0.5 <= gabarito <= 0.8
    assert abs(dif_baixo) < 0.01, (gabarito, dif_baixo)
