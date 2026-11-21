"""1.10(c): o comportamento de Ashkrof (environment/comportamento.py)."""
import math

import numpy as np
import pytest

from environment.comportamento import ComportamentoAshkrof, Contexto
from environment.env import GradeCelulas, GradeQuadrada, Veiculo
from tests.test_comportamento import PrefFixa
from tests.test_regras import MoveUmaVez, ZONA, pedido_no_proximo_passo, simulador_vazio

GRADE3 = GradeQuadrada(3)  # a zona 4 é o centro; vizinhas na ordem [1, 7, 3, 5, 0, 2, 6, 8]
PREF = PrefFixa({4: 0.05, 1: 0.10, 7: 0.25, 3: 0.20, 5: 0.15, 0: 0.10, 2: 0.05, 6: 0.05, 8: 0.05})


class TemposFixos:
    def __init__(self, minutos):
        self.valor = minutos

    def minutos(self, a, b, hora):
        return self.valor


def motorista(zona=4, base=4, corridas=0, fim=0):
    v = Veiculo(0, zona)
    v.base, v.corridas_feitas, v.fim_ultima_corrida = base, corridas, fim
    return v


def contexto(tempo=1, minutos=5.0, tipo_dia="util", grade=GRADE3, pref=PREF):
    return Contexto(hora=9, preferencias=pref, grade=grade, tipo_dia=tipo_dia, tempo=tempo,
                    tempos=TemposFixos(minutos))


ASH = ComportamentoAshkrof()


def probs(**kw):
    rec = kw.pop("rec", 1)
    ctx_kw = {k: kw.pop(k) for k in ("tempo", "minutos", "tipo_dia") if k in kw}
    return ASH.probabilidades(motorista(**kw), rec, contexto(**ctx_kw))


def test_utilidades_e_probabilidades_pela_formula():
    # espera 2 passos = 20 min, 3 corridas, 6 min até a recomendada, base 8 (vizinha de 4: familiar), sábado
    v, ctx = motorista(zona=4, base=8, corridas=3, fim=0), contexto(tempo=2, minutos=6.0, tipo_dia="sabado")
    v_w, v_h, v_c = ASH.utilidades(v, 0, ctx)
    assert v_w == pytest.approx(0.207 - 0.022 * 20 + 0.322 + 0.350)
    assert v_h == pytest.approx(0.080 * 3 - 0.025 * 6)
    assert v_c == pytest.approx(-0.312)
    p = ASH.probabilidades(v, 0, ctx)
    total = sum(math.exp(x) for x in (v_w, v_h, v_c))
    assert p == pytest.approx(tuple(math.exp(x) / total for x in (v_w, v_h, v_c)))
    assert sum(p) == pytest.approx(1)
    assert ASH.utilidades(v, 4, ctx)[1] == pytest.approx(0.080 * 3)  # ficar: 0 minutos (pendência A)


def test_tetos_de_corridas_e_espera():
    assert probs(corridas=30) == pytest.approx(probs(corridas=14))
    assert probs(tempo=9) == pytest.approx(probs(tempo=3))            # 90 min vale 25 (o mesmo que 30)
    assert probs(tempo=3) != pytest.approx(probs(tempo=2))            # 30 → 25 e 20 → 20 diferem


def test_monotonia():
    p_h = [probs(corridas=c)[1] for c in range(0, 20)]
    assert all(a < b for a, b in zip(p_h[:14], p_h[1:15])) and len(set(np.round(p_h[14:], 12))) == 1
    p_w = [probs(tempo=t)[0] for t in range(0, 4)]                    # espera 0, 10, 20, 30 min
    assert p_w[0] > p_w[1] > p_w[2] > p_w[3]
    familiar = ASH.probabilidades(motorista(zona=4, base=4), 1, contexto())[2]
    longe = ASH.probabilidades(motorista(zona=0, base=8), 1, contexto())[2]  # 8 não é vizinha de 0
    assert familiar < longe
    assert probs(minutos=2.0)[1] > probs(minutos=8.0)[1]


def test_decisao_por_faixa_da_acumulada():
    v, ctx = motorista(), contexto()
    p_w, p_h, p_c = ASH.probabilidades(v, 1, ctx)
    assert ASH.responder(v, 1, ctx, p_w / 2, 0.5) == (False, 4)                  # W: recusa e fica
    assert ASH.responder(v, 1, ctx, p_w + p_h / 2, 0.5) == (True, 1)            # H: aceita e vai
    aceitou, destino = ASH.responder(v, 1, ctx, 1 - p_c / 2, 0.5)               # C: recusa e sai
    assert aceitou is False and destino not in (4, 1) and destino in GRADE3.vizinhas(4)
    # ficar: W e H obedecem; C sai
    q_w, q_h, q_c = ASH.probabilidades(v, 4, ctx)
    assert ASH.responder(v, 4, ctx, q_w / 2, 0.5) == (True, 4)
    assert ASH.responder(v, 4, ctx, q_w + q_h / 2, 0.5) == (True, 4)
    assert ASH.responder(v, 4, ctx, 1 - q_c / 2, 0.5)[0] is False


def test_sair_rodando_nunca_volta_e_sem_candidata_fica():
    v, ctx = motorista(), contexto()
    destinos = {ASH.sair_rodando(v, 7, ctx, u) for u in np.linspace(0, 0.999, 200)}
    assert destinos <= set(GRADE3.vizinhas(4)) - {7}
    dupla = GradeCelulas([(0, 0), (1, 0)])  # a zona 0 só tem a vizinha 1, que é a recomendada
    v2 = motorista(zona=0, base=0)
    assert ASH.sair_rodando(v2, 1, contexto(grade=dupla, pref=PrefFixa({0: 0.5, 1: 0.5})), 0.3) == 0


def test_sem_recomendacao_usa_a_conduta_de_sempre():
    assert ASH.responder(motorista(), None, contexto(), 0.0, 0.0) == (None, 4)


def test_frequencias_de_w_h_c():
    v, ctx = motorista(base=8, corridas=5), contexto(tempo=2, minutos=4.0)
    p = ASH.probabilidades(v, 1, ctx)
    rng, n = np.random.default_rng(0), 30_000
    contagem = {"W": 0, "H": 0, "C": 0}
    for _ in range(n):
        aceitou, destino = ASH.responder(v, 1, ctx, rng.random(), rng.random())
        contagem["H" if aceitou else ("W" if destino == 4 else "C")] += 1
    for alt, pa in zip("WHC", p):
        assert abs(contagem[alt] / n - pa) < 4 * np.sqrt(pa * (1 - pa) / n), (alt, contagem[alt] / n, pa)


def test_estado_corridas_e_espera():
    sim = simulador_vazio()
    sim.veiculos[0].zona = ZONA
    pedido_no_proximo_passo(sim, destino=ZONA, duracao_min=25)   # 2 passos: termina em t + 2
    sim.rodar(3)
    v = sim.veiculos[0]
    assert (v.corridas_feitas, v.fim_ultima_corrida) == (1, 3)

    sim = simulador_vazio(MoveUmaVez(6))                           # um reposicionamento, sem corrida
    sim.rodar(3)
    v = sim.veiculos[0]
    assert v.zona == 6 and (v.corridas_feitas, v.fim_ultima_corrida) == (0, 0)  # a espera continua contando
