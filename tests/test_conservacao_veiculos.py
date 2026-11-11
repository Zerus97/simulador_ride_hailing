from environment.env import Grade
from simulator.simulator import Simulador
from algorithms.hotspot import Hotspot


def criar_simulador(seed=0, num_veiculos=10):
    return Simulador(Grade(5), num_veiculos=num_veiculos, lam=0.4,
                     algoritmo=Hotspot(), seed=seed)


def test_veiculos_ociosos_mais_em_corrida_e_a_frota():
    sim = criar_simulador()

    for _ in range(200):
        sim.passo()

        ociosos = sum(1 for v in sim.veiculos if v.ocioso)
        em_corrida = len(sim.corridas_ativas)
        assert ociosos + em_corrida == len(sim.veiculos), f"t={sim.tempo}"
