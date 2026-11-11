from environment.env import Grade
from simulator.simulator import Simulador
from algorithms.hotspot import Hotspot


def criar_simulador(seed=0, num_veiculos=10):
    return Simulador(Grade(5), num_veiculos=num_veiculos, lam=0.4,
                     algoritmo=Hotspot(), seed=seed)


def test_ids_do_historico_sao_sequenciais():
    sim = criar_simulador()

    sim.rodar(200)

    ids = [p.id for p in sim.historico_pedidos]
    assert len(ids) > 0
    assert ids == list(range(1, sim.total_pedidos + 1))
