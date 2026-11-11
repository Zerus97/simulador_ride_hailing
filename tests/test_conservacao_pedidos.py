from environment.env import Grade
from simulator.simulator import Simulador
from algorithms.hotspot import Hotspot


def criar_simulador(seed=0, num_veiculos=10):
    return Simulador(Grade(5), num_veiculos=num_veiculos, lam=0.4,
                     algoritmo=Hotspot(), seed=seed)


def test_pedidos_gerados_sao_casados_expirados_ou_abertos():
    sim = criar_simulador()

    for _ in range(200):
        sim.passo()

        casados = len(sim.corridas_ativas) + len(sim.corridas_concluidas)
        abertos = sum(sim.contar_pedidos_por_zona().values())
        assert sim.total_pedidos == casados + sim.pedidos_expirados + abertos, f"t={sim.tempo}"
