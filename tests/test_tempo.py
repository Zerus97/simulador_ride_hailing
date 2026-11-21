from pathlib import Path

import pandas as pd
import pytest

from environment.tempo import (MINUTOS_POR_PASSO, PASSOS_POR_DIA, SEGUNDOS_POR_PASSO,
                               espera_min, passos_da_corrida)


def test_constantes():
    assert (SEGUNDOS_POR_PASSO, MINUTOS_POR_PASSO, PASSOS_POR_DIA) == (600, 10, 144)


@pytest.mark.parametrize("duracao_min, passos", [
    (0, 1),       # mínimo de 1
    (4, 1),
    (14, 1),
    (None, 1),    # Poisson: 1 passo, como hoje
    (15, 1),      # para baixo: floor(1,5) = 1
    (19.9, 1),
    (20, 2),
    (25, 2),      # floor(2,5) = 2
    (30, 3),
    (3.17, 1),
])
def test_passos_da_corrida(duracao_min, passos):
    assert passos_da_corrida(duracao_min) == passos


def segundo(h, m, s=0):
    return h * 3600 + m * 60 + s


@pytest.mark.parametrize("pedido, passos_ate_atender, minutos", [
    (segundo(3, 7), 0, 3.0),          # 3h07, atendido no mesmo passo: espera até 3h10
    (segundo(3, 7), 1, 13.0),         # 3h07, atendido no passo seguinte: até 3h20
    (segundo(3, 0), 0, 10.0),         # 3h00:00 em ponto, mesmo passo
    (segundo(3, 9, 59), 0, 1 / 60),   # 3h09:59, mesmo passo: 1 segundo
])
def test_espera_min(pedido, passos_ate_atender, minutos):
    assert espera_min(pedido, passos_ate_atender) == pytest.approx(minutos)


# --- Com os dados reais (pula sem o dataset) ---

DATASET = Path(__file__).resolve().parent.parent / "dataset"


@pytest.mark.parametrize("dia", [23, 19])
def test_vies_do_arredondamento_fica_abaixo_de_0_3_min(dia):
    # o quanto, em média, o simulador deixa o carro ocupado além do real
    arquivo = DATASET / "pedidos_1200.parquet"
    if not arquivo.exists():
        pytest.skip("dataset fora do repositório")
    pedidos = pd.read_parquet(arquivo, columns=["dia", "duracao_min"])
    duracoes = pedidos.loc[pedidos["dia"] == dia, "duracao_min"]

    simulado = duracoes.map(passos_da_corrida) * MINUTOS_POR_PASSO
    vies = simulado.mean() - duracoes.mean()

    assert abs(vies) < 0.3, vies
