import pytest

from environment.env import Grade
from simulator.simulator import Simulador
from algorithms.hotspot import Hotspot
from algorithms.sem_movimento import SemMovimento


ALGORITMOS = [SemMovimento, Hotspot]
SEMENTES = [0, 1, 2]


def criar_simulador(algoritmo, seed, num_veiculos=10):
    return Simulador(Grade(5), num_veiculos=num_veiculos, lam=0.4,
                     algoritmo=algoritmo(), seed=seed)


@pytest.mark.parametrize("algoritmo", ALGORITMOS)
@pytest.mark.parametrize("seed", SEMENTES)
def test_pedidos_gerados_sao_casados_expirados_ou_abertos(algoritmo, seed):
    sim = criar_simulador(algoritmo, seed)

    for _ in range(200):
        sim.passo()

        casados = len(sim.corridas_ativas) + len(sim.corridas_concluidas)
        abertos = sum(sim.contar_pedidos_por_zona().values())
        assert sim.total_pedidos == casados + sim.pedidos_expirados + abertos, f"t={sim.tempo}"


@pytest.mark.parametrize("algoritmo", ALGORITMOS)
@pytest.mark.parametrize("seed", SEMENTES)
def test_veiculos_ociosos_mais_em_corrida_e_a_frota(algoritmo, seed):
    sim = criar_simulador(algoritmo, seed)

    for _ in range(200):
        sim.passo()

        ociosos = sum(1 for v in sim.veiculos if v.ocioso)
        em_corrida = len(sim.corridas_ativas)
        assert ociosos + em_corrida == len(sim.veiculos), f"t={sim.tempo}"


@pytest.mark.parametrize("algoritmo", ALGORITMOS)
@pytest.mark.parametrize("seed", SEMENTES)
def test_nenhum_pedido_aberto_esta_expirado(algoritmo, seed):
    sim = criar_simulador(algoritmo, seed)

    for _ in range(200):
        sim.passo()

        for linha in sim.pedidos:
            for pedidos_zona in linha:
                for pedido in pedidos_zona:
                    assert pedido.expira > sim.tempo, f"t={sim.tempo} {pedido}"


@pytest.mark.parametrize("algoritmo", ALGORITMOS)
@pytest.mark.parametrize("seed", SEMENTES)
def test_ids_do_historico_sao_sequenciais(algoritmo, seed):
    sim = criar_simulador(algoritmo, seed)

    sim.rodar(200)

    ids = [p.id for p in sim.historico_pedidos]
    assert len(ids) > 0
    assert ids == list(range(1, sim.total_pedidos + 1))
