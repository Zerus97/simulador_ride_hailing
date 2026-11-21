"""1.8(b): preferências dos motoristas (environment/preferencias.py)."""
import math
from pathlib import Path

import numpy as np
import pytest

from algorithms.sem_movimento import SemMovimento
from environment.env import GradeCelulas, grade_do_csv, nao_vizinhos_do_csv
from environment.preferencias import DIAS_POR_TIPO, preferencias_do_csv
from etl.etl import DIAS_TREINO, SABADOS_TREINO
from simulator.demanda import DemandaPoisson
from simulator.simulator import Simulador

DATASET = Path(__file__).resolve().parent.parent / "dataset"
# 3 × 1 células, 1 km de lado: zona 0 = (0,0), 1 = (1,0), 2 = (2,0); vizinhas 0–1 e 1–2
GRADE = GradeCelulas([(0, 0), (1, 0), (2, 0)])
L = math.log(2)


def csv_lambda(tmp_path, lam_hora):
    """lam_hora(zona, hora) -> λ da hora; vai todo no 1º passo da hora (a soma é o que conta)."""
    linhas = ["tipo_dia,passo,linha,coluna,pedidos_medios"]
    for passo in range(144):
        for z in GRADE.zonas():
            linha, coluna = GRADE.celula(z)
            valor = lam_hora(z, passo // 6) if passo % 6 == 0 else 0.0
            linhas.append(f"util,{passo},{linha},{coluna},{valor}")
    arquivo = tmp_path / "lambda.csv"
    arquivo.write_text("\n".join(linhas) + "\n")
    return arquivo


# λ por hora: 1,75; 3,75; 0. Com n_dias = 2, D = λ + 0,25 = 2, 4 e 0,25 (log: L, 2L, −2L)
def padrao(z, h):
    return (1.75, 3.75, 0.0)[z]


def alfa_a_mao():
    # numerador: média, sobre as zonas (todas as horas iguais), do desvio-padrão de log D
    sd_z0 = L / 2                       # candidatas 0, 1: (L, 2L)
    sd_z1 = L * math.sqrt(78 / 27)      # candidatas 1, 0, 2: (2L, L, −2L)
    sd_z2 = 2 * L                       # candidatas 2, 1: (−2L, 2L)
    numerador = (sd_z0 + sd_z1 + sd_z2) / 3
    # denominador: bases pesadas pela demanda do dia (1,75 : 3,75 : 0)
    w0, w1 = 1.75 / 5.5, 3.75 / 5.5
    z1 = w0 * math.sqrt(2 / 3) + w1 * math.sqrt(2 / 9)  # distâncias (1,0,2) e (0,1,1)
    denominador = (0.5 + z1 + 0.5) / 3                   # nas zonas 0 e 2, sempre 0,5
    return numerador / denominador


def test_alfa_pela_formula(tmp_path):
    pref = preferencias_do_csv(csv_lambda(tmp_path, padrao), GRADE, 1000, "util", n_dias=2)
    assert pref.alfa == pytest.approx(alfa_a_mao())


def test_probabilidades_conta_a_mao(tmp_path):
    pref = preferencias_do_csv(csv_lambda(tmp_path, padrao), GRADE, 1000, "util", n_dias=2)
    a = alfa_a_mao()
    p = pref.de(base=0, zona_atual=1, hora=9)
    # candidatas 1, 0, 2: D = 4, 2, 0,25; distância à base 0 = 1, 0, 2 km
    pesos = {1: 4 * math.exp(-a), 0: 2.0, 2: 0.25 * math.exp(-2 * a)}
    total = sum(pesos.values())
    assert set(p) == {0, 1, 2}
    assert p == pytest.approx({z: w / total for z, w in pesos.items()})
    assert sum(p.values()) == pytest.approx(1)
    assert set(pref.de(base=0, zona_atual=0, hora=9)) == {0, 1}  # só a atual e as vizinhas


def test_correcao_de_meio_pedido(tmp_path):
    # n_dias = 7: a zona 0 teve 1 pedido em 7 dias, a 2 nenhum; as duas a 1 km da base 1
    lam = csv_lambda(tmp_path, lambda z, h: (1 / 7, 1.0, 0.0)[z] if h else (1.0, 2.0, 3.0)[z])
    p = preferencias_do_csv(lam, GRADE, 1000, "util", n_dias=7).de(base=1, zona_atual=1, hora=5)
    assert p[2] > 0
    assert p[0] / p[2] == pytest.approx((1 / 7 + 0.5 / 7) / (0.5 / 7))  # = 3


def test_candidatas_todas_em_zero_sai_so_pela_distancia(tmp_path):
    # hora 0 sem nenhum pedido; nas outras, a demanda varia (α > 0)
    pref = preferencias_do_csv(csv_lambda(tmp_path, lambda z, h: padrao(z, h) if h else 0.0),
                               GRADE, 1000, "util", n_dias=2)
    p = pref.de(base=0, zona_atual=1, hora=0)
    pesos = {1: math.exp(-pref.alfa), 0: 1.0, 2: math.exp(-2 * pref.alfa)}
    assert pref.alfa > 0
    assert p == pytest.approx({z: w / sum(pesos.values()) for z, w in pesos.items()})


def test_base_puxa_com_demanda_igual(tmp_path):
    pref = preferencias_do_csv(csv_lambda(tmp_path, lambda z, h: 1.0 if h else padrao(z, h)),
                               GRADE, 1000, "util", n_dias=2)
    p = pref.de(base=2, zona_atual=1, hora=10)  # na hora 10, demanda igual nas três
    assert p[2] > p[1] > p[0]


def test_variacao_sem_demanda(tmp_path):
    pref = preferencias_do_csv(csv_lambda(tmp_path, padrao), GRADE, 1000, "util", n_dias=2,
                               peso_demanda=0.0)
    p = pref.de(base=0, zona_atual=1, hora=9)
    pesos = {1: math.exp(-pref.alfa), 0: 1.0, 2: math.exp(-2 * pref.alfa)}
    assert p == pytest.approx({z: w / sum(pesos.values()) for z, w in pesos.items()})


def test_dias_por_tipo_bate_com_o_etl():
    assert DIAS_POR_TIPO == {"util": len(DIAS_TREINO), "sabado": len(SABADOS_TREINO)}


def test_observacao_tem_as_preferencias_dos_ociosos(tmp_path):
    vistas = []

    class Guarda(SemMovimento):
        def decidir(self, obs):
            vistas.append(obs)
            return {}

    pref = preferencias_do_csv(csv_lambda(tmp_path, padrao), GRADE, 1000, "util", n_dias=2)
    sim = Simulador(GRADE, num_veiculos=6, demanda=DemandaPoisson(0.5, passo_inicial=18),
                    algoritmo=Guarda(), seed=0, preferencias=pref)
    sim.rodar(30)

    for obs in vistas:
        assert set(obs.preferencias) == {vid for vid, _ in obs.ociosos}
        for vid, zona in obs.ociosos:
            base = sim.veiculos[vid].base
            assert obs.preferencias[vid] == pref.de(base, zona, obs.passo_do_dia // 6)


# --- diagnóstico com os dados (pula sem o dataset) ---

def quantis_ponderados(valores, pesos, qs):
    ordem = np.argsort(valores)
    v, w = np.asarray(valores)[ordem], np.asarray(pesos)[ordem]
    acumulado = np.cumsum(w) / w.sum()
    return [float(v[np.searchsorted(acumulado, q)]) for q in qs]


def test_diagnostico_das_preferencias_em_manhattan(capsys):
    arquivos = [DATASET / n for n in ("grade_1200.csv", "nao_vizinhos_1200.csv", "lambda_1200.csv")]
    if not all(a.exists() for a in arquivos):
        pytest.skip("dataset fora do repositório")
    grade = grade_do_csv(arquivos[0], nao_vizinhos=nao_vizinhos_do_csv(arquivos[1]))
    pref = preferencias_do_csv(arquivos[2], grade, 1200, "util", DIAS_POR_TIPO["util"])

    linhas = [f"α = {pref.alfa:.4f} por km (1,2 km, dias úteis)"]
    for hora in (5, 9, 18):
        preferida, ficar, pesos = [], [], []
        for z in grade.zonas():  # cada zona com o mesmo peso; bases pela demanda
            for b, w in pref.pesos_base.items():
                if w > 0:
                    p = pref.de(b, z, hora)
                    preferida.append(max(p.values()))
                    ficar.append(p[z])
                    pesos.append(w)
        for nome, valores in (("zona preferida", preferida), ("ficar", ficar)):
            p10, p50, p90 = quantis_ponderados(valores, pesos, (0.1, 0.5, 0.9))
            linhas.append(f"{hora:>2}h, {nome:<14}: p10 {p10:.3f} | mediana {p50:.3f} | p90 {p90:.3f}")
            assert 0 < p10 <= p50 <= p90 <= 1
    with capsys.disabled():
        print("\n" + "\n".join(linhas))
