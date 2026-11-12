from environment.env import GradeQuadrada
from simulator.simulator import Simulador
from algorithms.hotspot import Hotspot
from algorithms.sem_movimento import SemMovimento


def criar_simulador_1(seed=0, num_veiculos=10):
    return Simulador(GradeQuadrada(5), num_veiculos=num_veiculos, lam=0.4,
                     algoritmo=Hotspot(), seed=seed)

def criar_simulador_2(seed=0, num_veiculos=10):
    return Simulador(GradeQuadrada(5), num_veiculos=num_veiculos, lam=0.4,
                     algoritmo=SemMovimento(), seed=seed)


def test_algoritmo_nao_muda_os_pedidos_gerados():
    a = criar_simulador_1(seed=0)
    b = criar_simulador_2(seed=0)

    a.rodar(200)
    b.rodar(200)

    pedidos_a = [(p.criado_em, p.origem, p.destino) for p in a.historico_pedidos]
    pedidos_b = [(p.criado_em, p.origem, p.destino) for p in b.historico_pedidos]
    
    assert pedidos_a == pedidos_b and len(pedidos_a) > 0