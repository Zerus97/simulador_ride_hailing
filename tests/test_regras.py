"""Regras do simulador (seções da proposta), testadas sobre o código atual.

Os exemplos explícitos usam lam=0 (sem demanda aleatória), SemMovimento e
pedidos e veículos colocados à mão, para controlar cada passo.
Ordem dentro de passo(): tempo += 1, demanda, chegadas, expiração, matching,
reposicionamento.
"""
import pytest

from algorithms.algoritmo_reposicionamento import AlgoritmoReposicionamento
from algorithms.hotspot import Hotspot
from algorithms.sem_movimento import SemMovimento
from environment.env import GradeQuadrada, Pedido
from simulator.simulator import Simulador

ALGORITMOS = [SemMovimento, Hotspot]
SEMENTES = [0, 1, 2]

# Grade 5x5:  0 1 2 3 4 / 5 6 7 8 9 / 10 11 12 13 14 / ...
ZONA = 12
LONGE = 0  # não é vizinha de 12
DESTINO = 24


def simulador_vazio(algoritmo=None):
    """Um veículo, parado em LONGE, sem demanda aleatória."""
    sim = Simulador(GradeQuadrada(5), num_veiculos=1, lam=0,
                    algoritmo=algoritmo or SemMovimento(), seed=0)
    sim.veiculos[0].zona = LONGE
    return sim


def pedido_no_proximo_passo(sim, zona=ZONA, destino=DESTINO):
    """Cria um pedido como se a demanda o gerasse no próximo passo (t = tempo + 1)."""
    pedido = Pedido(sim.proximo_id_pedido, None, zona, destino, 1, sim.tempo + 1)
    sim.proximo_id_pedido += 1
    sim.total_pedidos += 1
    sim.pedidos[zona].append(pedido)
    sim.historico_pedidos.append(pedido)
    return pedido


def atendidos(sim):
    return [c.pedido for c in sim.corridas_ativas + sim.corridas_concluidas]


# --- §6 Expiração: criado em t, atendível em t e em t+1, expira em t+2 ---

def test_pedido_sem_veiculo_fica_aberto_em_t_e_t1_e_expira_em_t2():
    sim = simulador_vazio()
    pedido = pedido_no_proximo_passo(sim)
    t = pedido.criado_em

    sim.passo()
    assert sim.tempo == t and pedido in sim.pedidos[ZONA]
    sim.passo()
    assert sim.tempo == t + 1 and pedido in sim.pedidos[ZONA]
    sim.passo()
    assert sim.tempo == t + 2 and pedido not in sim.pedidos[ZONA]
    assert sim.pedidos_expirados == 1


@pytest.mark.parametrize("passos_antes, atendido", [
    (0, True),   # veículo chega à zona no passo t
    (1, True),   # ... no passo t+1
    (2, False),  # ... no passo t+2: o pedido já expirou
])
def test_pedido_e_atendivel_em_t_e_t1_mas_nao_em_t2(passos_antes, atendido):
    sim = simulador_vazio()
    pedido = pedido_no_proximo_passo(sim)
    for _ in range(passos_antes):
        sim.passo()

    sim.veiculos[0].zona = ZONA
    sim.passo()

    assert sim.tempo == pedido.criado_em + passos_antes
    assert (pedido in atendidos(sim)) == atendido


# --- §6 Ordem de chegada: na zona, o pedido mais antigo é atendido primeiro ---

def test_pedido_mais_antigo_da_zona_e_atendido_primeiro():
    sim = simulador_vazio()
    antigo = pedido_no_proximo_passo(sim)
    sim.passo()
    novo = pedido_no_proximo_passo(sim)
    assert novo.criado_em == antigo.criado_em + 1

    sim.veiculos[0].zona = ZONA
    sim.passo()

    assert atendidos(sim) == [antigo]
    assert list(sim.pedidos[ZONA]) == [novo]


# --- §4 Reposicionamento em 1 passo: movido em t, está na zona nova em t+1 ---

class MoveUmaVez(AlgoritmoReposicionamento):
    """Manda o veículo 0 para `destino` no primeiro passo e depois não mexe mais."""

    def __init__(self, destino):
        self.destino = destino
        self.ja_moveu = False

    def decidir(self, veiculos_ociosos, corridas_ativas, pedidos, grade):
        if self.ja_moveu:
            return {}
        self.ja_moveu = True
        return {0: self.destino}


def test_veiculo_movido_em_t_so_atende_na_zona_nova_em_t1():
    vizinha = 6  # vizinha de LONGE (0)
    sim = simulador_vazio(MoveUmaVez(vizinha))
    pedido = pedido_no_proximo_passo(sim, zona=vizinha)

    sim.passo()  # passo t: o matching vem antes do movimento
    assert sim.veiculos[0].zona == vizinha
    assert pedido not in atendidos(sim)

    sim.passo()  # passo t+1: já está na zona nova e atende
    assert pedido in atendidos(sim)


# --- §5 Recomendação válida: a zona atual ou uma vizinha ---

class Espiao(AlgoritmoReposicionamento):
    """Repassa a decisão do algoritmo real e guarda (zona atual, destino) de cada veículo."""

    def __init__(self, algoritmo):
        self.algoritmo = algoritmo
        self.recomendacoes = []

    def decidir(self, veiculos_ociosos, corridas_ativas, pedidos, grade):
        moves = self.algoritmo.decidir(veiculos_ociosos, corridas_ativas, pedidos, grade)
        zona_atual = dict(veiculos_ociosos)
        for vid, destino in moves.items():
            self.recomendacoes.append((vid, zona_atual.get(vid), destino))
        return moves


@pytest.mark.parametrize("algoritmo", ALGORITMOS)
@pytest.mark.parametrize("seed", SEMENTES)
def test_recomendacao_e_zona_atual_ou_vizinha(algoritmo, seed):
    espiao = Espiao(algoritmo())
    grade = GradeQuadrada(5)
    sim = Simulador(grade, num_veiculos=10, lam=0.4, algoritmo=espiao, seed=seed)

    sim.rodar(200)

    for vid, zona, destino in espiao.recomendacoes:
        assert zona is not None, f"veículo {vid} recebeu recomendação sem estar ocioso"
        if destino is not None:  # None = fica onde está
            assert destino == zona or destino in grade.vizinhas(zona), \
                f"veículo {vid}: {zona} -> {destino}"


# --- §10 Métricas coerentes: taxa de atendimento = atendidos ÷ total ---

@pytest.mark.parametrize("algoritmo", ALGORITMOS)
@pytest.mark.parametrize("seed", SEMENTES)
def test_taxa_de_atendimento_e_atendidos_sobre_total(algoritmo, seed):
    sim = Simulador(GradeQuadrada(5), num_veiculos=10, lam=0.4, algoritmo=algoritmo(), seed=seed)

    sim.rodar(200)
    m = sim.metrics()

    ids_atendidos = {p.id for p in atendidos(sim)}
    assert len(ids_atendidos) == len(atendidos(sim)), "pedido atendido mais de uma vez"
    assert m["taxa_atendimento"] == pytest.approx(len(ids_atendidos) / m["pedidos_totais"])

    abertos = len(sim.pedidos_abertos())
    assert m["taxa_atendimento"] + m["taxa_pedidos_expirados"] + abertos / m["pedidos_totais"] \
        == pytest.approx(1)
