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
    (15, 2),      # 1,5 arredonda para cima
    (24, 2),
    (25, 3),      # floor(2,5 + 0,5) = 3; o round() do Python daria 2
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
