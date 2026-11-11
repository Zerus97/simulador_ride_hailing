from environment.env import Grade
from simulator.simulator import Simulador
from algorithms.hotspot import Hotspot


def criar_simulador(seed=0, num_veiculos=10):
    return Simulador(Grade(5), num_veiculos=num_veiculos, lam=0.4,
                     algoritmo=Hotspot(), seed=seed)


def test_nenhum_pedido_aberto_esta_expirado():
    sim = criar_simulador()

    for _ in range(200):
        sim.passo()

        for linha in sim.pedidos:
            for pedidos_zona in linha:
                for pedido in pedidos_zona:
                    assert pedido.expira > sim.tempo, f"t={sim.tempo} {pedido}"
