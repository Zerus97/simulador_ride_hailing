import pytest

from environment.env import Grade
from simulator.simulator import Simulador
from algorithms.hotspot import Hotspot
from algorithms.sem_movimento import SemMovimento


# Fotografia das métricas com semente 0 e 500 passos (NumPy 2.5.1).
# Estes valores não são "os corretos": são os de hoje.
# Reorganização do código -> não podem mudar (se mudarem, é bug).
# Mudança de modelo -> atualize aqui e explique o motivo no commit.
@pytest.mark.parametrize("algoritmo, esperado", [
    (SemMovimento, {
        "taxa_atendimento": 0.3186502323701758,
        "pedidos_totais": 4949,
        "pedidos_expirados": 3352,
        "taxa_pedidos_expirados": 0.6773085471812488,
        "corridas_completadas": 1574,
        "taxa_ociosidade": 0.6846,
    }),
    (Hotspot, {
        "taxa_atendimento": 0.5136391190139422,
        "pedidos_totais": 4949,
        "pedidos_expirados": 2387,
        "taxa_pedidos_expirados": 0.4823196605374823,
        "corridas_completadas": 2537,
        "taxa_ociosidade": 0.4916,
    }),
])
def test_metricas_de_referencia(algoritmo, esperado):
    sim = Simulador(Grade(5), num_veiculos=10, lam=0.4,
                    algoritmo=algoritmo(), seed=0)

    sim.rodar(500)

    assert sim.metrics() == pytest.approx(esperado)
