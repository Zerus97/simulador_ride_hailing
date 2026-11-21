"""Regras do simulador (seções da proposta), testadas sobre o código atual.

Os exemplos explícitos usam DemandaPoisson(0) (sem demanda aleatória), SemMovimento e
pedidos e veículos colocados à mão, para controlar cada passo.
Ordem dentro de passo(): tempo += 1, demanda, chegadas, expiração, matching,
reposicionamento.
"""
import pytest

from algorithms.algoritmo_reposicionamento import AlgoritmoReposicionamento
from algorithms.hotspot import Hotspot
from algorithms.sem_movimento import SemMovimento
from environment.env import EM_CORRIDA, OCIOSO, REPOSICIONANDO, GradeQuadrada, Pedido
from simulator.demanda import DemandaPoisson
from simulator.simulator import Simulador

ALGORITMOS = [SemMovimento, Hotspot]
SEMENTES = [0, 1, 2]

# Grade 5x5:  0 1 2 3 4 / 5 6 7 8 9 / 10 11 12 13 14 / ...
ZONA = 12
LONGE = 0  # não é vizinha de 12
DESTINO = 24


def simulador_vazio(algoritmo=None):
    """Um veículo, parado em LONGE, sem demanda aleatória."""
    sim = Simulador(GradeQuadrada(5), num_veiculos=1, demanda=DemandaPoisson(0),
                    algoritmo=algoritmo or SemMovimento(), seed=0)
    sim.veiculos[0].zona = LONGE
    return sim


def pedido_no_proximo_passo(sim, zona=ZONA, destino=DESTINO, duracao_min=None, segundo_do_dia=None):
    """Cria um pedido como se a demanda o gerasse no próximo passo (t = tempo + 1)."""
    pedido = Pedido(sim.proximo_id_pedido, None, zona, destino, 1, sim.tempo + 1,
                    segundo_do_dia=segundo_do_dia, duracao_min=duracao_min)
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
    v = sim.veiculos[0]
    # no fim do passo t, o carro está a caminho: ainda na zona de origem
    assert (v.estado, v.destino, v.zona) == (REPOSICIONANDO, vizinha, LONGE)
    assert pedido not in atendidos(sim)

    sim.passo()  # passo t+1: já está na zona nova e atende
    assert pedido in atendidos(sim)


# --- 1.5: estados do veículo e corridas com a duração real ---

def test_corrida_longa_ocupa_o_carro_pela_duracao():
    # 25 min -> floor(2,5 + 0,5) = 3 passos: EM_CORRIDA em t, t+1 e t+2; OCIOSO em t+3
    sim = simulador_vazio()
    sim.veiculos[0].zona = ZONA
    pedido = pedido_no_proximo_passo(sim, duracao_min=25)
    t = pedido.criado_em

    estados = []
    for _ in range(4):
        sim.passo()
        estados.append((sim.tempo - t, sim.veiculos[0].estado))

    assert estados == [(0, EM_CORRIDA), (1, EM_CORRIDA), (2, EM_CORRIDA), (3, OCIOSO)]
    assert sim.veiculos[0].zona == DESTINO


def test_reposicionado_chega_ocioso_antes_do_matching():
    vizinha = 6  # vizinha de LONGE (0)
    sim = simulador_vazio(MoveUmaVez(vizinha))
    sim.passo()  # passo t: recomendação de mover
    assert sim.veiculos[0].estado == REPOSICIONANDO

    vistos = []
    matching = sim.matching

    def espiao():
        vistos.append((sim.veiculos[0].estado, sim.veiculos[0].zona))
        matching()

    sim.matching = espiao
    sim.passo()  # passo t+1
    assert vistos == [(OCIOSO, vizinha)]
    assert (sim.veiculos[0].destino, sim.veiculos[0].chegada) == (None, None)


def test_recomendacao_para_a_propria_zona_e_ficar():
    sim = simulador_vazio(MoveUmaVez(LONGE))
    sim.passo()
    v = sim.veiculos[0]
    assert (v.estado, v.zona, v.destino, v.chegada) == (OCIOSO, LONGE, None, None)


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
    sim = Simulador(grade, num_veiculos=10, demanda=DemandaPoisson(0.4), algoritmo=espiao, seed=seed)

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
    sim = Simulador(GradeQuadrada(5), num_veiculos=10, demanda=DemandaPoisson(0.4), algoritmo=algoritmo(), seed=seed)

    sim.rodar(200)
    m = sim.metrics()

    ids_atendidos = {p.id for p in atendidos(sim)}
    assert len(ids_atendidos) == len(atendidos(sim)), "pedido atendido mais de uma vez"
    assert m["taxa_atendimento"] == pytest.approx(len(ids_atendidos) / m["pedidos_totais"])

    abertos = len(sim.pedidos_abertos())
    assert m["taxa_atendimento"] + m["taxa_pedidos_expirados"] + abertos / m["pedidos_totais"] \
        == pytest.approx(1)


@pytest.mark.parametrize("passos_antes, espera", [
    (0, 3.0),    # pedido das 3h07, atendido no mesmo passo: até 3h10
    (1, 13.0),   # atendido no passo seguinte: até 3h20
])
def test_corrida_guarda_a_espera_em_minutos(passos_antes, espera):
    sim = simulador_vazio()
    pedido_no_proximo_passo(sim, segundo_do_dia=3 * 3600 + 7 * 60)
    for _ in range(passos_antes):
        sim.passo()
    sim.veiculos[0].zona = ZONA
    sim.passo()
    (corrida,) = sim.corridas_ativas
    assert corrida.espera_min == pytest.approx(espera)


def test_pedido_da_poisson_nao_tem_espera():
    sim = Simulador(GradeQuadrada(5), num_veiculos=10, demanda=DemandaPoisson(0.4),
                    algoritmo=SemMovimento(), seed=0)
    sim.rodar(20)
    corridas = sim.corridas_ativas + sim.corridas_concluidas
    assert corridas and all(c.espera_min is None for c in corridas)
