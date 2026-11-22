"""1.10(d): a logística do i-Rebalance (AceiteRecomendacao) e a renda esperada."""
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from algorithms.algoritmo_reposicionamento import AlgoritmoReposicionamento
from environment.comportamento import (COEFICIENTES_LOGISTICA, AceiteRecomendacao, Comportamento, Contexto,
                                       Ficar)
from environment.env import GradeQuadrada, Veiculo
from environment.previsao import PrevisaoConstante, PrevisaoDemanda
from environment.renda import RendaEsperada
from simulator.demanda import DemandaPoisson
from simulator.simulator import Simulador
from tests.test_comportamento import PrefFixa

DATASET = Path(__file__).resolve().parent.parent / "dataset"
GRADE3 = GradeQuadrada(3)  # candidatas da zona 4: [4, 1, 7, 3, 5, 0, 2, 6, 8]
# preferências: 7 > 3 > 5 > 1 > 0 > 4 > 2 > 6 > 8  (posições 1, 2, 3, 4, ...)
PREF = PrefFixa({7: 0.30, 3: 0.20, 5: 0.15, 1: 0.12, 0: 0.08, 4: 0.06, 2: 0.04, 6: 0.03, 8: 0.02})
B, WR, WM, WO = COEFICIENTES_LOGISTICA["reajuste"]


def previsao(valor_por_zona, passo=51):
    lam = {(z, p): 0.0 for z in GRADE3.zonas() for p in range(144)}
    lam.update({(z, passo): v for z, v in valor_por_zona.items()})
    return PrevisaoDemanda(lam)


def modelo(o=0.5, coeficientes="reajuste", deterministico=False):
    m = AceiteRecomendacao(deterministico=deterministico, coeficientes=coeficientes, niveis=[o], contagens=[1])
    m.iniciar([Veiculo(0, 4)], np.random.default_rng(0))
    return m


def contexto(lam=None, oferta=None, fator=None):
    return Contexto(hora=8, preferencias=PREF, grade=GRADE3, previsao=previsao(lam or {1: 2.0}),
                    passo_do_dia=50, renda=RendaEsperada(fator or {z: 1.0 for z in GRADE3.zonas()}),
                    oferta=dict(oferta or {}))


def sigma(x):
    return 1 / (1 + math.exp(-x))


def test_conta_a_mao_coeficientes_do_reajuste():
    # recomendada 1: posição 4; D = 2, S = 3 (sem o próprio motorista, que está na 4), fator 1,5
    fator = {z: 1.0 for z in GRADE3.zonas()} | {1: 1.5}
    ctx, v = contexto(lam={1: 2.0}, oferta={1: 3}, fator=fator), Veiculo(0, 4)
    m_esperado = 14.6 * 1.5 * 2.0 / 3
    assert modelo().m(v, 1, ctx) == pytest.approx(m_esperado)
    assert modelo(o=0.7).prob(v, 1, ctx) == pytest.approx(sigma(B + WR * 4 + WM * m_esperado + WO * 0.7))


def test_deterministico_segue_o_sinal_do_logit():
    v = Veiculo(0, 4)
    for o, lam, rec in [(1.0, {7: 3.0}, 7), (0.0, {8: 0.0}, 8), (0.5, {1: 1.0}, 1), (0.9, {3: 4.0}, 3)]:
        ctx = contexto(lam=lam)
        logit = modelo(o).logit(v, rec, ctx)
        assert modelo(o, deterministico=True).prob(v, rec, ctx) == (1.0 if logit >= 0 else 0.0), (o, rec, logit)
    # logit exatamente 0: a regra deles aceita (logit ≥ 0)
    zero = modelo(coeficientes=(0.0, 0.0, 0.0, 0.0), deterministico=True)
    assert zero.logit(v, 1, contexto()) == 0 and zero.prob(v, 1, contexto()) == 1.0
    # dos dois lados do zero
    assert modelo(1.0, deterministico=True).prob(v, 7, contexto(lam={7: 3.0})) == 1.0
    assert modelo(0.0, deterministico=True).prob(v, 8, contexto(lam={8: 0.0})) == 0.0


def test_bordas_de_m():
    v = Veiculo(0, 4)
    assert modelo().m(v, 1, contexto(lam={1: 0.0}, oferta={1: 5})) == 0.0                  # D = 0
    fator = {z: 1.0 for z in GRADE3.zonas()} | {1: 1.8}
    assert modelo().m(v, 1, contexto(lam={1: 2.0}, oferta={}, fator=fator)) == pytest.approx(14.6 * 1.8)  # S = 0
    # ficar: o próprio motorista não conta na oferta da zona atual (S = 3 − 1 = 2; contando, daria 3)
    assert modelo().m(v, 4, contexto(lam={4: 2.0}, oferta={4: 3})) == pytest.approx(14.6 * 2 / 2)


def test_monotonia_em_posicao_renda_e_obediencia():
    v = Veiculo(0, 4)
    iguais = {z: 2.0 for z in GRADE3.zonas()}
    ctx = contexto(lam=iguais)
    por_posicao = [modelo().prob(v, z, ctx) for z in (7, 3, 5, 1, 0)]       # posições 1 a 5
    assert all(a > b for a, b in zip(por_posicao, por_posicao[1:]))
    baixo = contexto(lam={1: 2.0}, oferta={1: 4}, fator={z: 1.0 for z in GRADE3.zonas()} | {1: 0.8})
    alto = contexto(lam={1: 2.0}, oferta={1: 4}, fator={z: 1.0 for z in GRADE3.zonas()} | {1: 1.6})
    assert modelo().prob(v, 1, alto) > modelo().prob(v, 1, baixo)
    assert modelo(o=0.9).prob(v, 1, ctx) > modelo(o=0.2).prob(v, 1, ctx)


def test_aglomeracao_no_mesmo_passo():
    """10 carros na zona 4, todos mandados para a 1; quem decide depois vê os anteriores."""
    vistos = []

    class Registra(AceiteRecomendacao):
        def m(self, veiculo, zona, ctx):
            valor = super().m(veiculo, zona, ctx)
            vistos.append(valor)
            return valor

    class TodosPara1(AlgoritmoReposicionamento):
        def decidir(self, obs):
            return {vid: 1 for vid, _ in obs.ociosos}

    sim = Simulador(GRADE3, num_veiculos=10, demanda=DemandaPoisson(0), algoritmo=TodosPara1(), seed=0,
                    previsao=PrevisaoConstante(2.0), preferencias=PREF,
                    renda=RendaEsperada({z: 1.0 for z in GRADE3.zonas()}),
                    comportamento=Comportamento(Registra(coeficientes=(10.0, 0.0, 0.0, 0.0)), Ficar()))
    for v in sim.veiculos:
        v.zona = 4
    sim.passo()

    assert len(vistos) == 10 and sim.ultimo_relatorio.aceites == tuple(range(10))
    assert vistos[0] == pytest.approx(14.6) and vistos[1] == pytest.approx(14.6)   # S = 0, 1 ≤ D = 2
    assert all(a >= b for a, b in zip(vistos, vistos[1:])) and vistos[-1] == pytest.approx(14.6 * 2 / 9)


# a ficha da 1.10d: (r, m, o) -> logit dos publicados, P dos publicados, P do código
TABELA = [
    (1, 14.6, 0.49, 3.5473, 0.9720, 0.9676),
    (5, 14.6, 0.49, 1.7873, 0.8566, 0.8629),
    (9, 14.6, 0.49, 0.0273, 0.5068, 0.5702),
    (3, 10.0, 0.20, 0.7040, 0.6691, 0.6761),
    (9, 6.0, 0.00, -3.5300, 0.0285, 0.0415),
    (1, 16.0, 1.00, 5.0600, 0.9937, 0.9914),
]


@pytest.mark.parametrize("r, m, o, logit_pub, p_pub, p_cod", TABELA)
def test_tabela_de_referencia_nos_dois_conjuntos(r, m, o, logit_pub, p_pub, p_cod):
    publicados, codigo = AceiteRecomendacao(), AceiteRecomendacao(coeficientes="codigo")
    assert publicados.logit_de(r, m, o) == pytest.approx(logit_pub, abs=1e-4)
    assert publicados.prob_de(r, m, o) == pytest.approx(p_pub, abs=1e-4)
    assert codigo.prob_de(r, m, o) == pytest.approx(p_cod, abs=1e-4)


@pytest.mark.parametrize("r, m, o, logit, decisao", [
    (9, 10, 0.5, -1.04775, 0.0),
    (1, 6, 0.5, 0.89305, 1.0),
    (5, 8, 0.5, -0.07735, 0.0),   # perto do zero, de propósito: um erro de sinal ou coeficiente troca
    (5, 9, 0.5, 0.21585, 1.0),
])
def test_fidelidade_ao_codigo_deles(r, m, o, logit, decisao):
    codigo = AceiteRecomendacao(coeficientes="codigo", deterministico=True)
    assert codigo.logit_de(r, m, o) == pytest.approx(logit, abs=1e-5)
    assert codigo.prob_de(r, m, o) == decisao


def test_padrao_e_os_publicados_e_desconhecido_da_erro():
    assert (AceiteRecomendacao().b, AceiteRecomendacao().w_o) == (-1.31, 2.17)
    with pytest.raises(ValueError, match="xyz"):
        AceiteRecomendacao(coeficientes="xyz")


def test_reajuste_bate_com_o_questionario():
    arquivo = DATASET / "questionaire.csv"
    if not arquivo.exists():
        pytest.skip("questionaire.csv fora do repositório")
    q = pd.read_csv(arquivo)
    q.columns = [c.strip() for c in q.columns]
    X = np.column_stack([np.ones(len(q)), q[["Ranking", "Revenue", "obey"]].to_numpy(float)])
    y = 1 - q["UnAccepted"].to_numpy(float)
    b = np.zeros(4)
    for _ in range(50):  # IRLS
        p = 1 / (1 + np.exp(-X @ b))
        b = b + np.linalg.solve(X.T @ ((p * (1 - p))[:, None] * X), X.T @ (y - p))
    assert tuple(b) == pytest.approx(COEFICIENTES_LOGISTICA["reajuste"], abs=1e-3)
    p = 1 / (1 + np.exp(-X @ b))
    erros = np.sqrt(np.diag(np.linalg.inv(X.T @ ((p * (1 - p))[:, None] * X))))
    assert tuple(erros) == pytest.approx((0.254, 0.025, 0.022, 0.210), abs=1e-3)
    assert (len(y), round(y.mean(), 3)) == (1854, 0.662)
