"""1.9(a): o módulo de comportamento (environment/comportamento.py)."""
from pathlib import Path

import numpy as np
import pytest

from algorithms.algoritmo_reposicionamento import AlgoritmoReposicionamento
from algorithms.hotspot import Hotspot
from algorithms.sem_movimento import SemMovimento
from environment.comportamento import (OBEDIENCIA_CONTAGENS, OBEDIENCIA_NIVEIS, AceiteFixo,
                                       Comportamento, ConfiancaQuestionario, Contexto, Ficar,
                                       Proporcional,
                                       candidatas)
from environment.env import REPOSICIONANDO, GradeQuadrada, Veiculo
from environment.preferencias import Preferencias
from simulator.demanda import DemandaPoisson
from simulator.simulator import Simulador

GRADE3 = GradeQuadrada(3)  # zonas 0..8; a 4 é o centro


def preferencias_de_mentira():
    """Preferências numa grade 3 × 3, com demanda por hora sorteada (fixa pela semente)."""
    rng = np.random.default_rng(0)
    lam = {(z, h): float(rng.uniform(0.5, 5)) for z in GRADE3.zonas() for h in range(24)}
    demanda = {k: v + 0.25 for k, v in lam.items()}
    dia = {z: sum(lam[(z, h)] for h in range(24)) for z in GRADE3.zonas()}
    return Preferencias(GRADE3, 1000, lam, demanda, {z: d / sum(dia.values()) for z, d in dia.items()})


PREF = preferencias_de_mentira()


class FicarTodos(AlgoritmoReposicionamento):
    """Recomenda ficar a todos os ociosos."""

    def decidir(self, obs):
        return {vid: zona for vid, zona in obs.ociosos}


class GuardaSaida(AlgoritmoReposicionamento):
    def __init__(self, algoritmo):
        self.algoritmo, self.saida = algoritmo, None

    def decidir(self, obs):
        self.saida = self.algoritmo.decidir(obs)
        return self.saida


def comportamento(rho, regra=None, sem="ficar"):
    return Comportamento(AceiteFixo(rho), regra or Ficar(), sem_recomendacao=sem)


def frequencias_batem(destinos, esperado):
    """Cada frequência a menos de 4 desvios-padrão da probabilidade esperada."""
    n = len(destinos)
    for zona, p in esperado.items():
        f = destinos.count(zona) / n
        assert abs(f - p) < 4 * np.sqrt(p * (1 - p) / n) + 1e-12, (zona, f, p)
    assert set(destinos) <= set(esperado)


# --- ρ = 1 e ρ = 0 ---

def test_rho_1_e_o_simulador_de_antes():
    def rodada(**kw):
        sim = Simulador(GradeQuadrada(5), num_veiculos=10, demanda=DemandaPoisson(0.4),
                        algoritmo=Hotspot(), seed=1, **kw)
        sim.rodar(200)
        return ([(p.origem, p.destino, p.criado_em) for p in sim.historico_pedidos],
                [v.zona for v in sim.veiculos], sim.metrics(), sim.ultimo_relatorio.recusas)

    assert rodada() == rodada(comportamento=comportamento(1.0))
    assert rodada()[3] == ()


def test_rho_0_com_ficar_ninguem_se_move():
    espiao = GuardaSaida(Hotspot())
    sim = Simulador(GradeQuadrada(5), num_veiculos=10, demanda=DemandaPoisson(0.4),
                    algoritmo=espiao, seed=0, comportamento=comportamento(0.0))
    for _ in range(100):
        sim.passo()
        assert not any(v.estado == REPOSICIONANDO for v in sim.veiculos)
        assert sim.ultimo_relatorio.recusas == tuple(sorted(espiao.saida))
        assert sim.ultimo_relatorio.aceites == ()


def test_frequencia_de_aceite_com_rho_0_3():
    aceites = total = 0
    for seed in range(20):
        sim = Simulador(GradeQuadrada(5), num_veiculos=1000, demanda=DemandaPoisson(0),
                        algoritmo=FicarTodos(), seed=seed, comportamento=comportamento(0.3))
        sim.passo()
        aceites += len(sim.ultimo_relatorio.aceites)
        total += len(sim.ultimo_relatorio.aceites) + len(sim.ultimo_relatorio.recusas)
    assert total == 20_000
    assert abs(aceites / total - 0.3) < 4 * np.sqrt(0.3 * 0.7 / total)


def test_aceite_encaixado_entre_valores_de_rho():
    def aceitos(rho):
        sim = Simulador(GradeQuadrada(5), num_veiculos=300, demanda=DemandaPoisson(0),
                        algoritmo=FicarTodos(), seed=5, comportamento=comportamento(rho))
        sim.passo()
        return set(sim.ultimo_relatorio.aceites)

    baixo, alto = aceitos(0.3), aceitos(0.7)
    assert baixo and baixo < alto


# --- regra proporcional e conduta sem recomendação ---

def destinos_sorteados(comport, recomendacao, n=20_000, seed=0):
    veiculo = Veiculo(0, 4)
    veiculo.base = 8
    ctx = Contexto(hora=9, preferencias=PREF, grade=GRADE3)
    rng = np.random.default_rng(seed)
    respostas = [comport.responder(veiculo, recomendacao, ctx, rng.random(), rng.random()) for _ in range(n)]
    return [d for _, d in respostas], {a for a, _ in respostas}


def test_proporcional_segue_as_preferencias():
    destinos, aceitou = destinos_sorteados(comportamento(0.0, Proporcional()), recomendacao=1)
    assert aceitou == {False}
    frequencias_batem(destinos, PREF.de(8, 4, 9))
    assert candidatas(4, GRADE3)[0] == 4  # a ordem fixa começa pela zona atual


def test_sem_recomendacao_ficar_ou_preferencias():
    destinos, aceitou = destinos_sorteados(comportamento(1.0, sem="ficar"), recomendacao=None, n=200)
    assert aceitou == {None} and set(destinos) == {4}
    destinos, aceitou = destinos_sorteados(comportamento(1.0, sem="preferencias"), recomendacao=None)
    assert aceitou == {None}
    frequencias_batem(destinos, PREF.de(8, 4, 9))


def test_recomendacao_de_ficar_recusada_pode_mover():
    moveu = 0
    for seed in range(50):
        sim = Simulador(GRADE3, num_veiculos=1, demanda=DemandaPoisson(0), algoritmo=FicarTodos(),
                        seed=seed, preferencias=PREF, comportamento=comportamento(0.0, Proporcional()))
        sim.passo()
        assert sim.ultimo_relatorio.recusas == (0,)
        moveu += sim.veiculos[0].estado == REPOSICIONANDO
    assert moveu > 0


@pytest.mark.parametrize("sem", ["ficar", "preferencias"])
def test_sem_recomendacao_fica_fora_de_aceites_e_recusas(sem):
    sim = Simulador(GRADE3, num_veiculos=20, demanda=DemandaPoisson(0), algoritmo=SemMovimento(),
                    seed=0, preferencias=PREF, comportamento=comportamento(0.5, sem=sem))
    sim.passo()
    assert sim.ultimo_relatorio.aceites == () and sim.ultimo_relatorio.recusas == ()
    movidos = sum(v.estado == REPOSICIONANDO for v in sim.veiculos)
    assert (movidos == 0) if sem == "ficar" else (movidos > 0)


# --- erros de configuração ---

@pytest.mark.parametrize("comport", [
    comportamento(0.5, Proporcional()),
    comportamento(1.0, Ficar(), sem="preferencias"),
], ids=["proporcional", "sem_recomendacao_preferencias"])
def test_erro_na_construcao_sem_preferencias(comport):
    with pytest.raises(ValueError):
        Simulador(GRADE3, num_veiculos=2, demanda=DemandaPoisson(0), algoritmo=SemMovimento(),
                  seed=0, comportamento=comport)


def test_parametros_invalidos():
    with pytest.raises(ValueError):
        AceiteFixo(1.5)
    with pytest.raises(ValueError):
        Comportamento(AceiteFixo(1.0), Ficar(), sem_recomendacao="sair_rodando")


# --- 1.9(b): confiança fixa do questionário (estrutura; a tabela real vem do questionaire.csv) ---

def com_confianca(niveis, contagens, n=10, seed=0, algoritmo=None, passos=0):
    sim = Simulador(GradeQuadrada(5), num_veiculos=n, demanda=DemandaPoisson(0.4),
                    algoritmo=algoritmo or FicarTodos(), seed=seed,
                    comportamento=Comportamento(ConfiancaQuestionario(niveis, contagens), Ficar()))
    sim.rodar(passos)
    return sim


def test_confianca_frequencias_da_tabela():
    niveis, contagens = [0.2, 0.5, 0.9], [10, 30, 60]
    sim = com_confianca(niveis, contagens, n=10_000)
    obed = list(sim.comportamento.modelo.obediencia.values())
    for nivel, c in zip(niveis, contagens):
        p = c / 100
        assert abs(obed.count(nivel) / 10_000 - p) < 4 * np.sqrt(p * (1 - p) / 10_000)


def test_confianca_fixa_por_motorista_e_reprodutivel():
    a = com_confianca([0.1, 0.6, 0.95], [1, 1, 1], seed=4, passos=50)
    b = com_confianca([0.1, 0.6, 0.95], [1, 1, 1], seed=4)
    assert a.comportamento.modelo.obediencia == b.comportamento.modelo.obediencia
    ctx = Contexto(hora=9, preferencias=None, grade=a.grade)
    for v in a.veiculos:
        assert a.comportamento.modelo.prob(v, v.zona, ctx) == a.comportamento.modelo.obediencia[v.id]


def test_confianca_extremos_0_nunca_aceita_e_1_sempre():
    sim = com_confianca([0.0, 1.0], [1, 1], n=40, seed=1)
    obed = sim.comportamento.modelo.obediencia
    aceitos, recusados = set(), set()
    for _ in range(100):
        sim.passo()
        aceitos.update(sim.ultimo_relatorio.aceites)
        recusados.update(sim.ultimo_relatorio.recusas)
    assert aceitos and recusados
    assert all(obed[v] == 1.0 for v in aceitos) and all(obed[v] == 0.0 for v in recusados)


def test_confianca_nao_muda_os_numeros_de_aceite():
    def rodada(comport):
        sim = Simulador(GradeQuadrada(5), num_veiculos=50, demanda=DemandaPoisson(0.4),
                        algoritmo=FicarTodos(), seed=7, comportamento=comport)
        sim.passo()
        return sim

    fixo = rodada(comportamento(0.5))
    questionario = rodada(Comportamento(ConfiancaQuestionario([0.3, 0.8], [1, 1]), Ficar()))
    assert fixo.rng_aceite.bit_generator.state == questionario.rng_aceite.bit_generator.state
    assert ([(p.origem, p.destino) for p in fixo.historico_pedidos]
            == [(p.origem, p.destino) for p in questionario.historico_pedidos])


def test_confianca_sem_iniciar_da_erro():
    modelo = ConfiancaQuestionario([0.5], [1])
    with pytest.raises(RuntimeError):
        modelo.prob(Veiculo(0, 0), 0, None)


# --- a tabela do questionário ---

def test_tabela_do_questionario():
    assert len(OBEDIENCIA_NIVEIS) == len(OBEDIENCIA_CONTAGENS) == 14
    assert sum(OBEDIENCIA_CONTAGENS) == 87
    media = sum(n * c for n, c in zip(OBEDIENCIA_NIVEIS, OBEDIENCIA_CONTAGENS)) / 87
    assert media == pytest.approx(0.490, abs=5e-4)


def test_frequencias_do_questionario_com_8700_motoristas():
    sim = com_confianca(OBEDIENCIA_NIVEIS, OBEDIENCIA_CONTAGENS, n=8700)
    obed = list(sim.comportamento.modelo.obediencia.values())
    for nivel, c in zip(OBEDIENCIA_NIVEIS, OBEDIENCIA_CONTAGENS):
        p = c / 87
        assert abs(obed.count(nivel) / 8700 - p) < 4 * np.sqrt(p * (1 - p) / 8700), nivel
    assert ConfiancaQuestionario().niveis == list(OBEDIENCIA_NIVEIS)  # a tabela é o padrão


def test_tabela_bate_com_o_questionaire_csv():
    arquivo = Path(__file__).resolve().parent.parent / "dataset" / "questionaire.csv"
    if not arquivo.exists():
        pytest.skip("questionaire.csv fora do repositório")
    import pandas as pd
    q = pd.read_csv(arquivo)
    q.columns = [c.strip() for c in q.columns]
    assert (q.groupby("DriverID")["obey"].nunique() == 1).all()  # um valor por motorista
    contagem = q.groupby("DriverID")["obey"].first().value_counts().sort_index()
    assert tuple(contagem.index) == pytest.approx(OBEDIENCIA_NIVEIS)
    assert tuple(contagem.values) == OBEDIENCIA_CONTAGENS
