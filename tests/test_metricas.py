"""1.12(a): o coletor das métricas da proposta (simulator/metricas.py)."""
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from algorithms.hotspot import Hotspot
from algorithms.sem_movimento import SemMovimento
from algorithms.sem_recomendacao import SemRecomendacao
from environment.comportamento import AceiteFixo, Comportamento, Ficar, Proporcional
from environment.env import GradeQuadrada, grade_do_csv
from environment.tempos import TemposEntreZonas
from simulator.demanda import DemandaPoisson, DemandaRegistros
from simulator.simulator import Simulador
from tests.test_comportamento import FicarTodos
from tests.test_regras import LONGE, PREF_5X5, ZONA, MoveUmaVez, pedido_no_proximo_passo

DATASET = Path(__file__).resolve().parent.parent / "dataset"


def um_carro(algoritmo=None, aquecimento=0, **kw):
    """Um veículo na ZONA, sem demanda aleatória."""
    sim = Simulador(GradeQuadrada(5), num_veiculos=1, demanda=DemandaPoisson(0),
                    algoritmo=algoritmo or SemMovimento(), seed=0, passos_aquecimento=aquecimento, **kw)
    sim.veiculos[0].zona = ZONA
    return sim


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_atendimento_bate_com_a_metrica_antiga(seed):
    sim = Simulador(GradeQuadrada(5), num_veiculos=10, demanda=DemandaPoisson(0.4),
                    algoritmo=Hotspot(), seed=seed, passos_aquecimento=0)
    sim.rodar(200)
    m = sim.metricas()
    assert m["atendimento"] == pytest.approx(sim.metrics()["taxa_atendimento"])
    # com Poisson: não se aplica, e não zero
    assert m["espera_media_min"] is None and m["receita_usd"] is None
    assert m["reposicionamento_min"] is None and m["reposicionamento_km"] is None


def test_aquecimento_fica_fora():
    # destino = a própria ZONA: a corrida dura 1 passo e o carro volta a ficar livre lá
    sim = um_carro(aquecimento=6)
    sim.rodar(2)
    pedido_no_proximo_passo(sim, destino=ZONA)   # criado no passo 3: no aquecimento
    sim.rodar(3)
    pedido_no_proximo_passo(sim, destino=ZONA)   # criado no passo 6: ainda no aquecimento
    sim.passo()
    pedido_no_proximo_passo(sim, destino=ZONA)   # criado no passo 7: entra
    sim.rodar(3)

    m = sim.metricas()
    assert len(sim.corridas_ativas) + len(sim.corridas_concluidas) == 3  # os três foram atendidos
    assert (m["pedidos_gerados"], m["pedidos_atendidos"], m["atendimento"]) == (1, 1, 1.0)


def test_espera_em_minutos():
    sim = um_carro()
    pedido_no_proximo_passo(sim, segundo_do_dia=3 * 3600 + 7 * 60)  # 3h07, atendido no próprio passo
    sim.passo()
    assert sim.metricas()["espera_media_min"] == pytest.approx(3.0)


def test_receita():
    sim = um_carro()
    pedido_no_proximo_passo(sim, km=2.0, segundo_do_dia=0)
    sim.passo()
    sim.passo()                           # o carro chega e fica livre de novo
    sim.veiculos[0].zona = ZONA
    pedido_no_proximo_passo(sim, km=3.5, segundo_do_dia=1200)
    sim.passo()
    m = sim.metricas()
    assert m["pedidos_atendidos"] == 2
    assert m["receita_usd"] == pytest.approx(3.25 * (2.0 + 3.5))


def test_esforco_de_um_reposicionamento_recomendado():
    vizinha = 6  # vizinha de LONGE (0)
    tempos = TemposEntreZonas({(LONGE, vizinha): 2.0}, {(LONGE, vizinha): 1.5}, {h: 3.0 for h in range(24)})
    sim = Simulador(GradeQuadrada(5), num_veiculos=1, demanda=DemandaPoisson(0), algoritmo=MoveUmaVez(vizinha),
                    seed=0, tempos=tempos, passos_aquecimento=0)
    sim.veiculos[0].zona = LONGE
    sim.rodar(3)

    assert sim.coletor.movimentos[0][:4] == (0, LONGE, vizinha, "recomendado")
    m = sim.metricas()
    assert (m["reposicionamentos"], m["reposicionamento_min"], m["reposicionamento_km"]) == (1, 6.0, 1.5)


def tipos_de_movimento(algoritmo, comportamento):
    sim = Simulador(GradeQuadrada(5), num_veiculos=30, demanda=DemandaPoisson(0.2), algoritmo=algoritmo,
                    seed=0, preferencias=PREF_5X5, comportamento=comportamento, passos_aquecimento=0)
    sim.rodar(20)
    return {m[3] for m in sim.coletor.movimentos}, sim.metricas()["reposicionamentos"]


def test_movimentos_pos_recusa_nao_entram_no_esforco():
    tipos, esforco = tipos_de_movimento(FicarTodos(), Comportamento(AceiteFixo(0.0), Proporcional()))
    assert tipos == {"pos_recusa"} and esforco == 0


def test_movimentos_proprios_nao_entram_no_esforco():
    tipos, esforco = tipos_de_movimento(
        SemRecomendacao(), Comportamento(AceiteFixo(1.0), Ficar(), sem_recomendacao="preferencias"))
    assert tipos == {"proprio"} and esforco == 0


# --- com os dados (pula sem o dataset) ---

def test_dia_23_atendimento_e_receita():
    pedidos_arq, grade_arq = DATASET / "pedidos_1200.parquet", DATASET / "grade_1200.csv"
    if not (pedidos_arq.exists() and grade_arq.exists()):
        pytest.skip("dataset fora do repositório")
    fonte = DemandaRegistros(pd.read_parquet(pedidos_arq), dia=23, fracao=0.15, passo_inicial=18)
    sim = Simulador(grade_do_csv(grade_arq), num_veiculos=2000, demanda=fonte, algoritmo=SemMovimento(), seed=0)
    sim.rodar(126)

    m = sim.metricas()
    assert 0.6 < m["atendimento"] < 0.8, m["atendimento"]   # o gabarito deu ~70% com o aquecimento dentro
    atendidas = [c for c in sim.corridas_ativas + sim.corridas_concluidas if c.pedido.criado_em > 6]
    assert m["pedidos_atendidos"] == len(atendidas)
    assert m["receita_usd"] == pytest.approx(3.25 * sum(c.pedido.km for c in atendidas))
    assert 0 < m["espera_media_min"] <= 20
