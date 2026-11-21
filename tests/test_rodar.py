"""1.13(b): o executor de cenários (experimentos/rodar.py). Usa os dados; pula sem o dataset."""
import json
from pathlib import Path

import pandas as pd
import pytest

from algorithms.gulosa import GulosaDemanda
from environment.comportamento import Comportamento, ConfiancaQuestionario, Proporcional
from environment.env import grade_do_csv, nao_vizinhos_do_csv
from environment.preferencias import preferencias_do_csv
from environment.previsao import previsao_do_csv
from environment.tempos import tempos_do_csv
from experimentos.cenario import ler_cenario
from experimentos.rodar import COLUNAS_PASSO, rodar_cenario
from simulator.demanda import DemandaRegistros
from simulator.simulator import Simulador

DATASET = Path(__file__).resolve().parent.parent / "dataset"
ARQUIVOS = ("pedidos_1200.parquet", "grade_1200.csv", "nao_vizinhos_1200.csv", "lambda_1200.csv",
            "tempos_1200.csv", "fatores_hora.csv")
PEQUENO = """
[cenario]
nome = "pequeno"
finalidade = "avaliacao"
lado = 1200
dia = 23
fracao = 0.15
fator_demanda = 1.0
frota = 100
passo_inicial = 18
passos = 12
aquecimento = 6

[comportamento]
modelo = "questionario"
regra = "proporcional"
sem_recomendacao = "ficar"

[algoritmo]
nome = "gulosa"

[execucao]
sementes = [0, 1]
saida = "resultados/pequeno"
"""
DECISAO = ["decisao_media_s", "decisao_max_s"]


@pytest.fixture(scope="module")
def pedidos():
    if not all((DATASET / a).exists() for a in ARQUIVOS):
        pytest.skip("dataset fora do repositório")
    return pd.read_parquet(DATASET / ARQUIVOS[0])


@pytest.fixture
def cenario(tmp_path):
    caminho = tmp_path / "pequeno.toml"
    caminho.write_text(PEQUENO)
    return ler_cenario(caminho)


def test_arquivos_e_colunas(pedidos, cenario, tmp_path):
    resumo = rodar_cenario(cenario, pedidos, pasta=tmp_path / "saida")
    pasta = tmp_path / "saida"

    assert list(pd.read_csv(pasta / "resumo.csv")["semente"]) == [0, 1]
    assert {"atendimento", "espera_media_min", "receita_usd", "reposicionamentos", "aceite",
            "reposicionamento_sucesso", "ganhos_usd_por_veiculo", "frac_parado"} <= set(resumo.columns)
    for s in (0, 1):
        passos = pd.read_csv(pasta / f"passos_s{s}.csv")
        assert list(passos.columns) == COLUNAS_PASSO and len(passos) == 12
        assert list(passos["passo_do_dia"]) == list(range(18, 30))
        assert (passos[["ociosos", "em_corrida", "reposicionando"]].sum(axis=1) == 100).all()
    execucao = json.loads((pasta / "execucao.json").read_text())
    assert execucao["cenario"]["nome"] == "pequeno"
    assert {"commit", "alteracoes_nao_commitadas", "python", "numpy", "data"} <= set(execucao)


def test_mesma_semente_mesmo_resumo(pedidos, cenario, tmp_path):
    a = rodar_cenario(cenario, pedidos, pasta=tmp_path / "a").drop(columns=DECISAO)
    b = rodar_cenario(cenario, pedidos, pasta=tmp_path / "b").drop(columns=DECISAO)
    pd.testing.assert_frame_equal(a, b)


def test_executor_igual_ao_simulador_montado_a_mao(pedidos, cenario, tmp_path):
    resumo = rodar_cenario(cenario, pedidos, pasta=tmp_path / "saida")

    grade = grade_do_csv(DATASET / "grade_1200.csv", nao_vizinhos=nao_vizinhos_do_csv(DATASET / "nao_vizinhos_1200.csv"))
    sim = Simulador(
        grade, 100, DemandaRegistros(pedidos, 23, 0.15, 18), GulosaDemanda(), seed=1,
        previsao=previsao_do_csv(DATASET / "lambda_1200.csv", grade, "util", 0.15, 1.0),
        tempos=tempos_do_csv(DATASET / "tempos_1200.csv", DATASET / "fatores_hora.csv", grade, 1200, "util"),
        preferencias=preferencias_do_csv(DATASET / "lambda_1200.csv", grade, 1200, "util", 7),
        comportamento=Comportamento(ConfiancaQuestionario(), Proporcional(), sem_recomendacao="ficar"),
        passos_aquecimento=6)
    sim.rodar(12)

    a_mao = {k: v for k, v in sim.metricas().items() if k not in DECISAO}
    executor = resumo[resumo["semente"] == 1].drop(columns=["semente", *DECISAO]).iloc[0].to_dict()
    assert executor == pytest.approx(a_mao, nan_ok=True)
    assert a_mao["reposicionamentos"] > 0  # o cenário exercita os movimentos
