from environment.env import GradeQuadrada
from simulator.simulator import Simulador
from algorithms.hotspot import Hotspot


def criar_simulador(seed=0, num_veiculos=10):
    return Simulador(GradeQuadrada(5), num_veiculos=num_veiculos, lam=0.4,
                     algoritmo=Hotspot(), seed=seed)


def test_semente_diferente_da_diferente_metricas():
    a = criar_simulador(seed=0)
    b = criar_simulador(seed=1)

    a.rodar(200)
    b.rodar(200)

    assert a.metrics() != b.metrics()