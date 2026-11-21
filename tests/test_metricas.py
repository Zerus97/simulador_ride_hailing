"""1.12(a): o coletor das métricas da proposta (simulator/metricas.py)."""
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from algorithms.algoritmo_reposicionamento import AlgoritmoReposicionamento
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


# --- 1.12(b): as secundárias ---

def test_aceite_com_rho_0_3():
    sim = Simulador(GradeQuadrada(5), num_veiculos=500, demanda=DemandaPoisson(0.2), algoritmo=FicarTodos(),
                    seed=0, comportamento=Comportamento(AceiteFixo(0.3), Ficar()), passos_aquecimento=0)
    sim.rodar(20)
    c = sim.coletor
    n = c.aceites + c.recusas
    assert n > 5000
    assert abs(sim.metricas()["aceite"] - 0.3) < 4 * np.sqrt(0.3 * 0.7 / n)


def sucesso_de_um_movimento(pedido_em=None, passos=3):
    """O carro 0 é mandado de LONGE (0) para a vizinha 6 no passo 1 e chega no 2.
    pedido_em: passo em que aparece um pedido na zona 6 (None = nenhum)."""
    sim = Simulador(GradeQuadrada(5), num_veiculos=1, demanda=DemandaPoisson(0), algoritmo=MoveUmaVez(6),
                    seed=0, passos_aquecimento=0)
    sim.veiculos[0].zona = LONGE
    for t in range(1, passos + 1):
        if t == pedido_em:
            pedido_no_proximo_passo(sim, zona=6, destino=6)
        sim.passo()
    m = sim.metricas()
    return m["reposicionamentos_avaliados"], m["reposicionamento_sucesso"]


def test_sucesso_com_pedido_esperando_na_chegada():
    assert sucesso_de_um_movimento(pedido_em=1) == (1, 1.0)  # pedido desde o passo 1, carro chega no 2


def test_fracasso_em_zona_vazia():
    assert sucesso_de_um_movimento(pedido_em=None) == (1, 0.0)


def test_pedido_depois_da_chegada_e_fracasso():
    # a janela é fixa: só o matching de t + 1 conta
    assert sucesso_de_um_movimento(pedido_em=3) == (1, 0.0)


def test_movimento_do_ultimo_passo_fica_fora():
    avaliados, sucesso = sucesso_de_um_movimento(passos=1)
    assert avaliados == 0 and math.isnan(sucesso)


def test_ganhos_do_motorista():
    # carro 1 faz uma corrida de 2 km (3,25 × 2 = 6,50); carro 0 se move 1,5 km (0,23 × 1,5 = 0,345)
    tempos = TemposEntreZonas({(LONGE, 6): 2.0}, {(LONGE, 6): 1.5}, {h: 3.0 for h in range(24)})
    sim = Simulador(GradeQuadrada(5), num_veiculos=2, demanda=DemandaPoisson(0), algoritmo=MoveUmaVez(6),
                    seed=0, tempos=tempos, passos_aquecimento=0)
    sim.veiculos[0].zona, sim.veiculos[1].zona = LONGE, ZONA
    pedido_no_proximo_passo(sim, zona=ZONA, destino=ZONA, km=2.0, segundo_do_dia=0)
    sim.rodar(2)
    assert sim.metricas()["ganhos_usd_por_veiculo"] == pytest.approx((6.50 - 0.345) / 2)


def test_ociosidade_soma_1():
    sim = Simulador(GradeQuadrada(5), num_veiculos=10, demanda=DemandaPoisson(0.4), algoritmo=Hotspot(), seed=0)
    sim.rodar(100)
    m = sim.metricas()
    assert m["frac_reposicionando"] > 0 and m["frac_em_corrida"] > 0
    assert m["frac_parado"] + m["frac_reposicionando"] + m["frac_em_corrida"] == pytest.approx(1)


def test_tempo_de_decisao():
    class Lento(AlgoritmoReposicionamento):
        def decidir(self, obs):
            time.sleep(0.01)
            return {}

    sim = Simulador(GradeQuadrada(5), num_veiculos=2, demanda=DemandaPoisson(0), algoritmo=Lento(),
                    seed=0, passos_aquecimento=0)
    sim.rodar(3)
    m = sim.metricas()
    assert m["decisao_media_s"] >= 0.01 and m["decisao_max_s"] >= m["decisao_media_s"]



def test_aceite_separado_em_mover_e_ficar():
    from algorithms.algoritmo_reposicionamento import AlgoritmoReposicionamento
    from environment.comportamento import ModeloAceite

    class ParesFicamImparesMovem(AlgoritmoReposicionamento):
        def decidir(self, obs):
            return {vid: (zona if vid % 2 == 0 else obs.grade.vizinhas(zona)[0]) for vid, zona in obs.ociosos}

    class AceitaSoFicar(ModeloAceite):
        def prob(self, veiculo, zona_recomendada, ctx):
            return 1.0 if zona_recomendada == veiculo.zona else 0.0

    sim = Simulador(GradeQuadrada(5), num_veiculos=20, demanda=DemandaPoisson(0.3), algoritmo=ParesFicamImparesMovem(),
                    seed=0, comportamento=Comportamento(AceitaSoFicar(), Ficar()), passos_aquecimento=0)
    sim.rodar(30)
    m = sim.metricas()
    assert (m["aceite_ficar"], m["aceite_mover"]) == (1.0, 0.0)
    c = sim.coletor
    assert m["aceite"] == pytest.approx(sum(c.por_tipo["ficar"]) / (sum(c.por_tipo["ficar"]) + sum(c.por_tipo["mover"])))
    assert sum(c.por_tipo["mover"]) > 0 and sum(c.por_tipo["ficar"]) > 0


def test_aceite_so_de_ficar_deixa_o_de_mover_indefinido():
    sim = Simulador(GradeQuadrada(5), num_veiculos=100, demanda=DemandaPoisson(0.2), algoritmo=FicarTodos(), seed=0,
                    comportamento=Comportamento(AceiteFixo(0.3), Ficar()), passos_aquecimento=0)
    sim.rodar(10)
    m = sim.metricas()
    assert m["aceite_ficar"] == pytest.approx(m["aceite"]) and math.isnan(m["aceite_mover"])
