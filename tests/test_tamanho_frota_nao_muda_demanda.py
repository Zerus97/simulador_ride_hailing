from environment.env import Grade
from simulator.simulator import Simulador
from algorithms.hotspot import Hotspot


def criar_simulador_1(seed=0, num_veiculos=10):
    return Simulador(Grade(5), num_veiculos=num_veiculos, lam=0.4,
                     algoritmo=Hotspot(), seed=seed)

def criar_simulador_2(seed=0, num_veiculos=20):
    return Simulador(Grade(5), num_veiculos=num_veiculos, lam=0.4,
                     algoritmo=Hotspot(), seed=seed)


def test_mesma_semente_da_mesmas_metricas():
    a = criar_simulador_1(seed=0)
    b = criar_simulador_2(seed=0)

    a.rodar(200)
    b.rodar(200)

    assert a.historico_pedidos == b.historico_pedidos