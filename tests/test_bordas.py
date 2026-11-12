import math

from environment.env import GradeQuadrada
from simulator.simulator import Simulador
from algorithms.hotspot import Hotspot


def criar_simulador(seed=0, num_veiculos=10, lam=0.4):
    return Simulador(GradeQuadrada(5), num_veiculos=num_veiculos, lam=lam,
                     algoritmo=Hotspot(), seed=seed)


def test_zero_passos_nao_da_erro_e_taxas_sao_nan():
    sim = criar_simulador()

    sim.rodar(0)
    m = sim.metrics()

    assert math.isnan(m["taxa_atendimento"])
    assert math.isnan(m["taxa_pedidos_expirados"])
    assert math.isnan(m["taxa_ociosidade"])


def test_lam_zero_nao_da_erro_e_taxas_de_pedidos_sao_nan():
    sim = criar_simulador(lam=0)

    sim.rodar(200)
    m = sim.metrics()

    assert m["pedidos_totais"] == 0
    assert math.isnan(m["taxa_atendimento"])
    assert math.isnan(m["taxa_pedidos_expirados"])
    # sem pedidos, a frota fica ociosa o tempo todo
    assert m["taxa_ociosidade"] == 1.0
