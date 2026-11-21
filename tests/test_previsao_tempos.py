"""Previsão λ e tempos entre vizinhas (environment/previsao.py e environment/tempos.py)."""
from pathlib import Path

import pytest

from algorithms.contrato import Observacao
from algorithms.sem_movimento import SemMovimento
from environment.env import GradeCelulas, grade_do_csv, nao_vizinhos_do_csv
from environment.previsao import PrevisaoConstante, previsao_do_csv
from environment.tempos import tempos_do_csv
from simulator.demanda import DemandaPoisson
from simulator.simulator import Simulador

DATASET = Path(__file__).resolve().parent.parent / "dataset"
# grade de duas células: (0, 0) = zona 0, (0, 1) = zona 1 (vizinhas)
GRADE = GradeCelulas([(0, 0), (0, 1)])


def escrever(caminho, cabecalho, linhas):
    caminho.write_text(cabecalho + "\n" + "\n".join(",".join(map(str, l)) for l in linhas) + "\n")
    return caminho


# --- previsão ---

def csv_lambda(tmp_path, extra=()):
    """Tabela densa (2 células × 144 passos × 2 tipos), com valores conhecidos nos passos 0 e 1."""
    linhas = []
    for tipo, base in (("util", 10.0), ("sabado", 100.0)):
        for passo in range(144):
            for linha, coluna in map(GRADE.celula, GRADE.zonas()):
                valor = base + passo + coluna if passo < 2 else 0.0
                linhas.append((tipo, passo, linha, coluna, valor))
    return escrever(tmp_path / "lambda.csv", "tipo_dia,passo,linha,coluna,pedidos_medios",
                    linhas + list(extra))


def test_previsao_multiplica_por_fracao_e_fator(tmp_path):
    prev = previsao_do_csv(csv_lambda(tmp_path), GRADE, "util", fracao=0.15, fator=1.5)
    assert prev.lam(0, 0) == pytest.approx(10.0 * 0.15 * 1.5)   # célula (0, 0), passo 0
    assert prev.lam(1, 1) == pytest.approx(12.0 * 0.15 * 1.5)   # célula (0, 1), passo 1
    sabado = previsao_do_csv(csv_lambda(tmp_path), GRADE, "sabado", fracao=0.15)
    assert sabado.lam(0, 1) == pytest.approx(101.0 * 0.15)


def test_previsao_passo_144_vira_passo_0(tmp_path):
    prev = previsao_do_csv(csv_lambda(tmp_path), GRADE, "util", fracao=1.0)
    assert prev.lam(1, 144) == prev.lam(1, 0) == pytest.approx(11.0)


def test_previsao_zona_desconhecida_e_celula_fora_da_grade_dao_erro(tmp_path):
    prev = previsao_do_csv(csv_lambda(tmp_path), GRADE, "util", fracao=1.0)
    with pytest.raises(KeyError):
        prev.lam(7, 0)  # a zona 7 não existe: erro, e não 0
    with pytest.raises(KeyError):
        previsao_do_csv(csv_lambda(tmp_path, extra=[("util", 0, 5, 5, 1.0)]), GRADE, "util", 1.0)


def test_previsao_incompleta_da_erro(tmp_path):
    arquivo = escrever(tmp_path / "lambda.csv", "tipo_dia,passo,linha,coluna,pedidos_medios",
                       [("util", 0, 0, 0, 1.0)])
    with pytest.raises(ValueError):
        previsao_do_csv(arquivo, GRADE, "util", fracao=1.0)


# --- tempos ---

def csv_fatores(tmp_path):
    linhas = [(1200, tipo, hora, 100, (2.0 if hora < 12 else 3.0), 9.9)
              for tipo in ("util", "sabado") for hora in range(24)]
    return escrever(tmp_path / "fatores.csv", "lado,tipo_dia,hora,viagens,fator,fator_geral", linhas)


def csv_tempos(tmp_path, linhas=((0, 0, 0, 1, 2.5, 1.4), (0, 1, 0, 0, 3.0, 1.6))):
    return escrever(tmp_path / "tempos.csv", "linha_a,coluna_a,linha_b,coluna_b,osm_min,km", linhas)


def test_tempos_osm_vezes_fator_da_hora(tmp_path):
    t = tempos_do_csv(csv_tempos(tmp_path), csv_fatores(tmp_path), GRADE, 1200, "util")
    assert t.minutos(0, 1, hora=3) == pytest.approx(2.5 * 2.0)
    assert t.minutos(0, 1, hora=18) == pytest.approx(2.5 * 3.0)
    assert t.minutos(1, 0, hora=18) == pytest.approx(3.0 * 3.0)  # mão única: sentidos diferentes
    assert t.km(0, 1) == pytest.approx(1.4)                      # km sem fator


def test_tempos_ficar_e_zero_e_nao_vizinho_da_erro(tmp_path):
    t = tempos_do_csv(csv_tempos(tmp_path), csv_fatores(tmp_path), GRADE, 1200, "util")
    assert t.minutos(0, 0, hora=18) == 0.0 and t.km(1, 1) == 0.0
    separadas = GradeCelulas([(0, 0), (0, 1), (0, 5)])  # a (0, 5) não é vizinha de ninguém
    t = tempos_do_csv(csv_tempos(tmp_path), csv_fatores(tmp_path), separadas, 1200, "util")
    with pytest.raises(KeyError):
        t.minutos(0, separadas.zona((0, 5)), hora=3)


def test_tempos_falta_uma_vizinha_da_erro(tmp_path):
    so_ida = csv_tempos(tmp_path, linhas=[(0, 0, 0, 1, 2.5, 1.4)])  # falta 1 -> 0
    with pytest.raises(ValueError):
        tempos_do_csv(so_ida, csv_fatores(tmp_path), GRADE, 1200, "util")


# --- observação ---

def test_observacao_tem_passo_do_dia_e_previsao():
    vistas = []

    class Guarda(SemMovimento):
        def decidir(self, obs):
            vistas.append(obs)
            return {}

    sim = Simulador(GradeCelulas([(0, 0), (0, 1)]), num_veiculos=2,
                    demanda=DemandaPoisson(0.4, passo_inicial=18), algoritmo=Guarda(), seed=0,
                    previsao=PrevisaoConstante(0.4))
    sim.rodar(3)

    assert [o.passo_do_dia for o in vistas] == [18, 19, 20]   # passo_inicial + tempo - 1
    assert all(o.previsao.lam(z, o.passo_do_dia) == 0.4 for o in vistas for z in (0, 1))
    assert all(o.tempos is None for o in vistas)
    assert Observacao.__dataclass_fields__["previsao"].default is None


# --- com os dados (pulam sem o dataset) ---

@pytest.fixture(scope="module")
def grade_1200():
    grade, pares = DATASET / "grade_1200.csv", DATASET / "nao_vizinhos_1200.csv"
    if not (grade.exists() and pares.exists()):
        pytest.skip("dataset fora do repositório")
    return grade_do_csv(grade, nao_vizinhos=nao_vizinhos_do_csv(pares))


@pytest.mark.parametrize("tipo_dia, pedidos_por_dia", [("util", 324_954), ("sabado", 373_684)])
def test_lambda_do_dia_bate_com_a_1_4b(grade_1200, tipo_dia, pedidos_por_dia):
    prev = previsao_do_csv(DATASET / "lambda_1200.csv", grade_1200, tipo_dia, fracao=0.15)
    total = sum(prev.lam(z, p) for z in grade_1200.zonas() for p in range(144))
    assert total / 0.15 == pytest.approx(pedidos_por_dia, abs=1)


def test_pico_de_lambda_as_18h50(grade_1200):
    prev = previsao_do_csv(DATASET / "lambda_1200.csv", grade_1200, "util", fracao=0.15)
    pico = sum(prev.lam(z, 18 * 6 + 5) for z in grade_1200.zonas())
    assert pico == pytest.approx(585, abs=1)


@pytest.mark.parametrize("tipo_dia", ["util", "sabado"])
def test_vizinhas_ficam_abaixo_de_15_min_em_toda_hora(grade_1200, tipo_dia):
    t = tempos_do_csv(DATASET / "tempos_1200.csv", DATASET / "fatores_hora.csv", grade_1200, 1200, tipo_dia)
    assert len(t.pares()) == (169 - 11) * 2 == 316
    pior = max(t.minutos(a, b, h) for a, b in t.pares() for h in range(24))
    assert pior < 15, pior
