"""Regras do simulador (seções da proposta), testadas sobre o código atual.

Os exemplos explícitos usam DemandaPoisson(0) (sem demanda aleatória), SemMovimento e
pedidos e veículos colocados à mão, para controlar cada passo.
Ordem dentro de passo(): tempo += 1, demanda, chegadas, expiração, matching,
reposicionamento.
"""
import dataclasses
from collections import Counter

import numpy as np
import pytest

from algorithms.algoritmo_reposicionamento import AlgoritmoReposicionamento
from algorithms.hotspot import Hotspot
from algorithms.gulosa import GulosaDemanda
from algorithms.sem_movimento import SemMovimento
from algorithms.sem_recomendacao import SemRecomendacao
from algorithms.vizinho_aleatorio import VizinhoAleatorio
from environment.comportamento import AceiteFixo, Comportamento, Ficar
from environment.preferencias import Preferencias
from environment.previsao import PrevisaoDemanda
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

    def decidir(self, obs):
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
    # 39 min -> floor(3,9) = 3 passos (para baixo): EM_CORRIDA em t, t+1 e t+2; OCIOSO em t+3
    sim = simulador_vazio()
    sim.veiculos[0].zona = ZONA
    pedido = pedido_no_proximo_passo(sim, duracao_min=39)
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

    def iniciar(self, rng):
        self.algoritmo.iniciar(rng)  # o gerador vai para o algoritmo real

    def decidir(self, obs):
        moves = self.algoritmo.decidir(obs)
        zona_atual = dict(obs.ociosos)
        for vid, destino in moves.items():
            self.recomendacoes.append((vid, zona_atual.get(vid), destino))
        return moves


def preferencias_5x5():
    """Preferências e previsão de mentira na grade 5 × 5 (demanda sorteada, semente fixa)."""
    rng = np.random.default_rng(0)
    grade = GradeQuadrada(5)
    lam = {(z, h): float(rng.uniform(0.5, 5)) for z in grade.zonas() for h in range(24)}
    dia = {z: sum(lam[(z, h)] for h in range(24)) for z in grade.zonas()}
    pref = Preferencias(grade, 1000, lam, {k: v + 0.25 for k, v in lam.items()},
                        {z: d / sum(dia.values()) for z, d in dia.items()})
    previsao = PrevisaoDemanda({(z, p): lam[(z, p // 6)] / 6 for z in grade.zonas() for p in range(144)})
    return pref, previsao


PREF_5X5, PREVISAO_5X5 = preferencias_5x5()

# cada baseline com o que ele exige do simulador
CONFIGURACOES = {
    "sem_movimento": (SemMovimento, {}),
    "hotspot": (Hotspot, {}),
    "sem_recomendacao": (SemRecomendacao, {
        "preferencias": PREF_5X5,
        "comportamento": Comportamento(AceiteFixo(1.0), Ficar(), sem_recomendacao="preferencias")}),
    "vizinho_aleatorio": (VizinhoAleatorio, {}),
    "gulosa": (GulosaDemanda, {"previsao": PREVISAO_5X5}),
}


@pytest.mark.parametrize("nome", CONFIGURACOES)
@pytest.mark.parametrize("seed", SEMENTES)
def test_recomendacao_e_zona_atual_ou_vizinha(nome, seed):
    algoritmo, extras = CONFIGURACOES[nome]
    espiao = Espiao(algoritmo())
    grade = GradeQuadrada(5)
    sim = Simulador(grade, num_veiculos=10, demanda=DemandaPoisson(0.4), algoritmo=espiao, seed=seed,
                    **extras)

    sim.rodar(200)

    for vid, zona, destino in espiao.recomendacoes:
        assert zona is not None, f"veículo {vid} recebeu recomendação sem estar ocioso"
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


# --- 1.6: matching com motorista sorteado ---

def test_sorteio_do_motorista_e_justo():
    # dois carros ociosos na mesma zona e um pedido: em 2.000 sementes, cada um atende
    # ~1.000 vezes (desvio-padrão ~22; 900 a 1.100 é ±4,5 desvios). Pela ordem de id, 2.000 a 0.
    grade = GradeQuadrada(5)
    vezes = {0: 0, 1: 0}
    for seed in range(2000):
        sim = Simulador(grade, num_veiculos=2, demanda=DemandaPoisson(0),
                        algoritmo=SemMovimento(), seed=seed)
        for v in sim.veiculos:
            v.zona = ZONA
        pedido_no_proximo_passo(sim)
        sim.passo()
        (corrida,) = sim.corridas_ativas
        vezes[corrida.veiculo.id] += 1
    assert all(900 <= n <= 1100 for n in vezes.values()), vezes


def pares_pedido_veiculo(sim):
    return [(c.pedido.id, c.veiculo.id) for c in sim.corridas_concluidas + sim.corridas_ativas]


def rodada_poisson(seed, trocar_rng_matching=False):
    sim = Simulador(GradeQuadrada(5), num_veiculos=10, demanda=DemandaPoisson(0.4),
                    algoritmo=Hotspot(), seed=seed)
    if trocar_rng_matching:
        sim.rng_matching = np.random.default_rng(999)
    sim.rodar(200)
    return sim


def test_mesma_semente_mesmos_pares_pedido_veiculo():
    assert pares_pedido_veiculo(rodada_poisson(3)) == pares_pedido_veiculo(rodada_poisson(3))


def test_gerador_do_matching_nao_muda_os_pedidos():
    normal, trocado = rodada_poisson(3), rodada_poisson(3, trocar_rng_matching=True)

    def historico(sim):
        return [(p.origem, p.destino, p.criado_em) for p in sim.historico_pedidos]

    assert historico(normal) == historico(trocado)
    # e a troca teve efeito: outros motoristas atenderam
    assert pares_pedido_veiculo(normal) != pares_pedido_veiculo(trocado)


# --- 1.7(a): a observação que o algoritmo recebe ---

class GuardaObservacao(AlgoritmoReposicionamento):
    """Guarda cada Observacao recebida e repassa a decisão a `algoritmo` (se houver);
    opcionalmente esvazia os pedidos abertos dela."""

    def __init__(self, esvaziar=False, algoritmo=None):
        self.obs = []
        self.esvaziar = esvaziar
        self.algoritmo = algoritmo

    def decidir(self, obs):
        self.obs.append(obs)
        if self.esvaziar:
            obs.pedidos_abertos.clear()
        return self.algoritmo.decidir(obs) if self.algoritmo else {}


def situacao_montada(algoritmo):
    """Passo 1, à mão: o carro 0 ocioso na zona 0; o carro 1 pega um pedido de 25 min
    (2 passos) na zona 12 para a 24; dois pedidos esperam na zona 18, sem carro."""
    sim = Simulador(GradeQuadrada(5), num_veiculos=2, demanda=DemandaPoisson(0),
                    algoritmo=algoritmo, seed=0)
    sim.veiculos[0].zona, sim.veiculos[1].zona = 0, 12
    pedido_no_proximo_passo(sim, zona=12, destino=24, duracao_min=25)
    pedido_no_proximo_passo(sim, zona=18, destino=0)
    pedido_no_proximo_passo(sim, zona=18, destino=1)
    sim.passo()
    return sim


def test_observacao_mostra_o_estado_certo():
    espiao = GuardaObservacao()
    sim = situacao_montada(espiao)

    (obs,) = espiao.obs
    assert obs.tempo == 1
    assert obs.ociosos == ((0, 0),)
    assert obs.pedidos_abertos == {z: (2 if z == 18 else 0) for z in range(25)}
    assert obs.chegadas == Counter({(24, 3): 1})  # o carro 1 chega à zona 24 em t + 2
    assert obs.grade is sim.grade


def test_algoritmo_nao_mexe_no_simulador_pela_observacao():
    espiao = GuardaObservacao(esvaziar=True)
    sim = situacao_montada(espiao)

    assert len(sim.pedidos[18]) == 2  # o algoritmo esvaziou a cópia, não a fila
    (obs,) = espiao.obs
    with pytest.raises(dataclasses.FrozenInstanceError):
        obs.ociosos = ()


# --- 1.7(b): o relatório do passo ---

def test_relatorio_chega_no_passo_seguinte():
    espiao = GuardaObservacao(algoritmo=Hotspot())
    sim = Simulador(GradeQuadrada(5), num_veiculos=10, demanda=DemandaPoisson(0.4),
                    algoritmo=espiao, seed=0)
    sim.rodar(50)

    assert espiao.obs[0].relatorio is None
    assert all(obs.relatorio.tempo == obs.tempo - 1 for obs in espiao.obs[1:])
    assert sim.ultimo_relatorio.tempo == 50  # o do último passo nunca chega ao algoritmo


def test_relatorio_casamento_e_chegada_da_corrida():
    # carro 0 ocioso na ZONA e um pedido de 25 min (2 passos) na mesma zona
    sim = simulador_vazio()
    sim.veiculos[0].zona = ZONA
    pedido = pedido_no_proximo_passo(sim, duracao_min=25)
    t = pedido.criado_em

    sim.passo()
    assert sim.ultimo_relatorio.tempo == t
    assert sim.ultimo_relatorio.casamentos == ((0, pedido.id, ZONA),)
    assert sim.ultimo_relatorio.chegadas == ()

    sim.passo()
    assert sim.ultimo_relatorio.chegadas == ()        # ainda em corrida em t + 1
    sim.passo()
    assert sim.ultimo_relatorio.chegadas == ((0, DESTINO),)   # termina em t + 2


def test_relatorio_aceite_e_chegada_do_reposicionamento():
    vizinha = 6  # vizinha de LONGE (0)
    sim = simulador_vazio(MoveUmaVez(vizinha))

    sim.passo()
    assert sim.ultimo_relatorio.aceites == (0,)
    assert sim.ultimo_relatorio.recusas == ()
    sim.passo()
    assert sim.ultimo_relatorio.chegadas == ((0, vizinha),)
    assert sim.ultimo_relatorio.aceites == ()


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_relatorios_nao_perdem_casamentos(seed):
    espiao = GuardaObservacao(algoritmo=Hotspot())
    sim = Simulador(GradeQuadrada(5), num_veiculos=10, demanda=DemandaPoisson(0.4),
                    algoritmo=espiao, seed=seed)
    sim.rodar(200)

    relatorios = [obs.relatorio for obs in espiao.obs[1:]] + [sim.ultimo_relatorio]
    casamentos = sum(len(r.casamentos) for r in relatorios)
    assert casamentos == len(sim.corridas_ativas) + len(sim.corridas_concluidas)


# --- 1.7(c): a saída do algoritmo ---

class Devolve(AlgoritmoReposicionamento):
    """Devolve sempre a mesma saída."""

    def __init__(self, saida):
        self.saida = saida

    def decidir(self, obs):
        return dict(self.saida)


@pytest.mark.parametrize("saida", [
    {1: 12},     # o carro 1 está em corrida
    {0: 2},      # a zona 2 fica a duas zonas da 0
    {99: 0},     # não existe veículo 99
    {0: None},   # None não é recomendação
], ids=["carro_em_corrida", "duas_zonas", "id_inexistente", "none"])
def test_recomendacao_invalida_da_erro(saida):
    with pytest.raises(ValueError):
        situacao_montada(Devolve(saida))  # carro 0 ocioso na zona 0; carro 1 em corrida


def test_ficar_e_sem_recomendacao():
    sim = Simulador(GradeQuadrada(5), num_veiculos=2, demanda=DemandaPoisson(0),
                    algoritmo=Devolve({0: 0}), seed=0)
    sim.veiculos[0].zona, sim.veiculos[1].zona = 0, 24
    sim.passo()

    assert [(v.estado, v.zona) for v in sim.veiculos] == [(OCIOSO, 0), (OCIOSO, 24)]
    assert sim.ultimo_relatorio.aceites == (0,)  # o 1 ficou sem recomendação


class GuardaSaida(AlgoritmoReposicionamento):
    def __init__(self, algoritmo):
        self.algoritmo = algoritmo
        self.pares = []

    def decidir(self, obs):
        saida = self.algoritmo.decidir(obs)
        self.pares.append((obs, saida))
        return saida


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_hotspot_recomenda_para_todos_os_ociosos(seed):
    espiao = GuardaSaida(Hotspot())
    sim = Simulador(GradeQuadrada(5), num_veiculos=10, demanda=DemandaPoisson(0.4),
                    algoritmo=espiao, seed=seed)
    sim.rodar(200)
    for obs, saida in espiao.pares:
        assert set(saida) == {vid for vid, _ in obs.ociosos}, obs.tempo


# --- 1.7(e): o gancho sequencial ---

def test_algoritmo_sequencial_e_recusado_na_construcao():
    class Sequencial(AlgoritmoReposicionamento):
        sequencial = True

    with pytest.raises(NotImplementedError, match="4.6"):
        Simulador(GradeQuadrada(5), num_veiculos=2, demanda=DemandaPoisson(0.4),
                  algoritmo=Sequencial(), seed=0)


@pytest.mark.parametrize("nome", CONFIGURACOES)
def test_algoritmos_existentes_decidem_em_lote(nome):
    assert CONFIGURACOES[nome][0].sequencial is False
