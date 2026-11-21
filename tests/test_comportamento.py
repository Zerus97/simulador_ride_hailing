"""1.9(a): o módulo de comportamento (environment/comportamento.py)."""
from pathlib import Path

import numpy as np
import pytest

from algorithms.contrato import RelatorioDoPasso

from algorithms.algoritmo_reposicionamento import AlgoritmoReposicionamento
from algorithms.hotspot import Hotspot
from algorithms.sem_movimento import SemMovimento
from algorithms.sem_recomendacao import SemRecomendacao
from environment.comportamento import (OBEDIENCIA_CONTAGENS, OBEDIENCIA_NIVEIS, AceiteFixo,
                                       Comportamento, ConfiancaEvolutiva, ConfiancaQuestionario, Contexto,
                                       Ficar,
                                       Proporcional, UniformeQuatroPreferidas, ZonaPreferida,
                                       candidatas)
from environment.env import REPOSICIONANDO, GradeCelulas, GradeQuadrada, Veiculo
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
    algoritmo = SemMovimento() if sem == "ficar" else SemRecomendacao()
    sim = Simulador(GRADE3, num_veiculos=20, demanda=DemandaPoisson(0), algoritmo=algoritmo,
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


# --- 1.10(a): zona preferida e uniforme entre as 4 preferidas ---

class PrefFixa:
    """Preferências dadas à mão: o mesmo {zona: probabilidade} para qualquer base e hora."""

    def __init__(self, probs):
        self.probs = probs

    def de(self, base, zona_atual, hora):
        return self.probs


# na grade 3 × 3, as candidatas da zona 4 são [4, 1, 7, 3, 5, 0, 2, 6, 8] (a ordem de DIRECOES)
PREF_EMPATE = PrefFixa({4: 0.05, 1: 0.05, 7: 0.25, 3: 0.25, 5: 0.15, 0: 0.10, 2: 0.10, 6: 0.03, 8: 0.02})


def destinos_com(regra, pref, grade=GRADE3, zona=4, n=20_000, seed=0):
    veiculo = Veiculo(0, zona)
    ctx = Contexto(hora=9, preferencias=pref, grade=grade)
    comport = comportamento(0.0, regra)
    rng = np.random.default_rng(seed)
    return [comport.responder(veiculo, zona, ctx, rng.random(), rng.random())[1] for _ in range(n)]


def test_zona_preferida_e_a_de_maior_preferencia():
    esperado = max(candidatas(4, GRADE3), key=lambda z: PREF.de(8, 4, 9)[z])
    veiculo = Veiculo(0, 4)
    veiculo.base = 8
    ctx = Contexto(hora=9, preferencias=PREF, grade=GRADE3)
    assert ZonaPreferida().destino(veiculo, 1, ctx, u=0.99) == esperado


def test_zona_preferida_empate_pela_ordem_das_candidatas():
    # 7 e 3 empatam em 0,25; o 7 vem antes nas candidatas
    assert set(destinos_com(ZonaPreferida(), PREF_EMPATE, n=50)) == {7}


def test_zona_preferida_nao_muda_entre_sementes():
    # destinos_com usa um veículo na zona 4 com base 4
    assert {d for seed in range(20) for d in destinos_com(ZonaPreferida(), PREF, n=5, seed=seed)} == \
           {max(candidatas(4, GRADE3), key=lambda z: PREF.de(4, 4, 9)[z])}


def test_quatro_preferidas_uniforme_entre_as_4_maiores():
    # as 4 maiores: 7 e 3 (0,25), 5 (0,15), e 0 e 2 empatam em 0,10 -> o 0 vem antes
    destinos = destinos_com(UniformeQuatroPreferidas(), PREF_EMPATE)
    assert set(destinos) == {7, 3, 5, 0}
    frequencias_batem(destinos, {7: 0.25, 3: 0.25, 5: 0.25, 0: 0.25})


def test_quatro_preferidas_com_3_candidatas_sorteia_entre_as_3():
    linha = GradeCelulas([(0, 0), (1, 0), (2, 0)])  # a zona 1 tem as vizinhas 0 e 2
    destinos = destinos_com(UniformeQuatroPreferidas(), PrefFixa({1: 0.5, 0: 0.3, 2: 0.2}), grade=linha, zona=1)
    frequencias_batem(destinos, {1: 1 / 3, 0: 1 / 3, 2: 1 / 3})


# --- 1.10(b): confiança evolutiva ---

def evolutiva(mu0, n0=10, n=1):
    """Modelo iniciado com todos os motoristas em μ₀."""
    modelo = ConfiancaEvolutiva(n0, niveis=[mu0], contagens=[1])
    modelo.iniciar([Veiculo(i, 0) for i in range(n)], np.random.default_rng(0))
    return modelo


def relatorio(tempo, aceites=(), recusas=(), casados=()):
    return RelatorioDoPasso(tempo=tempo, casamentos=tuple((v, 100 + v, 0) for v in casados), chegadas=(),
                            aceites=tuple(aceites), recusas=tuple(recusas))


def test_aritmetica_do_beta():
    m = evolutiva(0.5, n0=10)
    assert m.confianca(0) == pytest.approx(5 / 10)
    m.fim_do_passo(relatorio(1, aceites=[0]))
    m.fim_do_passo(relatorio(2, casados=[0]))          # aceitou em 1, casou em 2: sucesso
    assert m.confianca(0) == pytest.approx(6 / 11)

    m = evolutiva(0.5, n0=10)
    m.fim_do_passo(relatorio(1, aceites=[0]))
    m.fim_do_passo(relatorio(2))                       # não casou em 2: fracasso
    assert m.confianca(0) == pytest.approx(5 / 11)


def test_so_atualiza_quem_aceitou():
    m = evolutiva(0.5, n0=10, n=3)
    # 0 aceitou, 1 recusou, 2 sem recomendação; os três casam no passo seguinte
    m.fim_do_passo(relatorio(1, aceites=[0], recusas=[1]))
    m.fim_do_passo(relatorio(2, casados=[0, 1, 2]))
    assert m.confianca(0) == pytest.approx(6 / 11)
    assert (m.alfa[1], m.beta[1]) == (5, 5) and (m.alfa[2], m.beta[2]) == (5, 5)


def test_janela_atendido_so_em_t_mais_2_e_fracasso():
    m = evolutiva(0.5, n0=10)
    m.fim_do_passo(relatorio(1, aceites=[0]))
    m.fim_do_passo(relatorio(2))                       # t + 1: nada
    m.fim_do_passo(relatorio(3, casados=[0]))          # t + 2: tarde demais (e não aceitou em 2)
    assert (m.alfa[0], m.beta[0]) == (5, 6)


def test_recomendacao_de_ficar_aceita_conta_como_sucesso():
    # μ₀ = 1: aceita a recomendação de ficar no passo 1; um pedido aparece na zona no passo 2
    sim = Simulador(GradeQuadrada(5), num_veiculos=1, demanda=DemandaPoisson(0), algoritmo=FicarTodos(), seed=0,
                    comportamento=Comportamento(ConfiancaEvolutiva(10, niveis=[1.0], contagens=[1]), Ficar()))
    sim.passo()
    assert sim.ultimo_relatorio.aceites == (0,)
    from tests.test_regras import pedido_no_proximo_passo
    pedido_no_proximo_passo(sim, zona=sim.veiculos[0].zona, destino=sim.veiculos[0].zona)
    sim.passo()
    assert sim.ultimo_relatorio.casamentos and sim.comportamento.modelo.alfa[0] == 11


def test_extremos_mu0_0_nunca_muda_e_mu0_1_cai_no_primeiro_fracasso():
    zero = evolutiva(0.0, n0=10)
    for t in range(1, 30):
        zero.fim_do_passo(relatorio(t, recusas=[0], casados=[0] if t % 2 else []))
    assert zero.confianca(0) == 0.0

    um = evolutiva(1.0, n0=10)
    um.fim_do_passo(relatorio(1, aceites=[0]))
    um.fim_do_passo(relatorio(2, casados=[0], aceites=[0]))   # sucesso: continua 1
    assert um.confianca(0) == 1.0
    um.fim_do_passo(relatorio(3))                             # fracasso
    assert um.confianca(0) == pytest.approx(11 / 12)


def test_n0_enorme_e_a_versao_fixa():
    def rodada(modelo):
        sim = Simulador(GradeQuadrada(5), num_veiculos=30, demanda=DemandaPoisson(0.4), algoritmo=Hotspot(),
                        seed=4, comportamento=Comportamento(modelo, Ficar()))
        sim.rodar(200)
        return ([(p.origem, p.destino, p.criado_em) for p in sim.historico_pedidos],
                [(c.pedido.id, c.veiculo.id) for c in sim.corridas_concluidas + sim.corridas_ativas],
                [v.zona for v in sim.veiculos], sim.metrics())

    assert rodada(ConfiancaEvolutiva(1e12)) == rodada(ConfiancaQuestionario())


def test_confianca_da_frota_e_a_media():
    sim = Simulador(GradeQuadrada(5), num_veiculos=50, demanda=DemandaPoisson(0.4), algoritmo=FicarTodos(),
                    seed=1, comportamento=Comportamento(ConfiancaEvolutiva(10), Ficar()), passos_aquecimento=0)
    sim.rodar(40)
    m = sim.comportamento.modelo
    media = np.mean([m.alfa[v] / (m.alfa[v] + m.beta[v]) for v in m.alfa])
    assert sim.metricas()["confianca_frota"] == pytest.approx(media)
    assert media != pytest.approx(np.mean(list(m.obediencia.values())))  # a confiança se moveu
    fixo = Simulador(GradeQuadrada(5), num_veiculos=5, demanda=DemandaPoisson(0.4), algoritmo=FicarTodos(), seed=1)
    fixo.rodar(5)
    assert fixo.metricas()["confianca_frota"] is None
