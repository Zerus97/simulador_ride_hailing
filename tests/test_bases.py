"""1.8(a): zonas-base dos motoristas e o gerador próprio do algoritmo."""
from pathlib import Path

import numpy as np
import pytest

from algorithms.algoritmo_reposicionamento import AlgoritmoReposicionamento
from algorithms.hotspot import Hotspot
from environment.env import GradeQuadrada, grade_do_csv, nao_vizinhos_do_csv
from environment.previsao import PrevisaoDemanda, previsao_do_csv
from simulator.demanda import DemandaPoisson
from simulator.simulator import Simulador

DATASET = Path(__file__).resolve().parent.parent / "dataset"


def test_bases_seguem_a_demanda():
    # λ três vezes maior na zona A (0) que na B (24), zero nas outras
    grade = GradeQuadrada(5)
    lam = {(z, p): (3.0 if z == 0 else 1.0 if z == 24 else 0.0) for z in grade.zonas() for p in range(144)}
    sim = Simulador(grade, num_veiculos=4000, demanda=DemandaPoisson(0), algoritmo=Hotspot(),
                    seed=0, previsao=PrevisaoDemanda(lam))

    bases = [v.base for v in sim.veiculos]
    assert set(bases) <= {0, 24}  # nenhum carro nas zonas de λ zero
    # 0,75 ± 4 desvios-padrão (sqrt(0,75 × 0,25 / 4000) ≈ 0,0068)
    assert abs(bases.count(0) / 4000 - 0.75) < 0.027


def test_base_e_a_zona_inicial_e_nao_muda():
    sim = Simulador(GradeQuadrada(5), num_veiculos=10, demanda=DemandaPoisson(0.4),
                    algoritmo=Hotspot(), seed=0)
    bases = [v.base for v in sim.veiculos]
    assert [v.zona for v in sim.veiculos] == bases

    sim.rodar(200)
    assert [v.base for v in sim.veiculos] == bases
    assert [v.zona for v in sim.veiculos] != bases  # o Hotspot moveu os carros


class Sorteia(AlgoritmoReposicionamento):
    """Não move ninguém, mas sorteia com o próprio gerador a cada passo."""

    def __init__(self):
        self.sorteios = []

    def decidir(self, obs):
        self.sorteios.extend(self.rng.random(5))
        return {}


def rodada(algoritmo, seed=3):
    sim = Simulador(GradeQuadrada(5), num_veiculos=10, demanda=DemandaPoisson(0.4),
                    algoritmo=algoritmo, seed=seed)
    sim.rodar(100)
    return sim


def test_algoritmo_recebe_gerador_proprio_e_reprodutivel():
    a, b = Sorteia(), Sorteia()
    rodada(a), rodada(b)
    assert isinstance(a.rng, np.random.Generator)
    assert a.sorteios == b.sorteios and len(a.sorteios) == 500


def test_gerador_do_algoritmo_nao_muda_os_pedidos():
    class Parado(AlgoritmoReposicionamento):
        def decidir(self, obs):
            return {}

    def historico(sim):
        return [(p.origem, p.destino, p.criado_em) for p in sim.historico_pedidos]

    assert historico(rodada(Sorteia())) == historico(rodada(Parado()))


# --- com os dados (pula sem o dataset) ---

def test_bases_em_manhattan_seguem_a_demanda_dos_dias_uteis():
    arquivos = [DATASET / n for n in ("grade_1200.csv", "nao_vizinhos_1200.csv", "lambda_1200.csv")]
    if not all(a.exists() for a in arquivos):
        pytest.skip("dataset fora do repositório")
    grade = grade_do_csv(arquivos[0], nao_vizinhos=nao_vizinhos_do_csv(arquivos[1]))
    previsao = previsao_do_csv(arquivos[2], grade, "util", fracao=0.15)
    n = 2000
    sim = Simulador(grade, num_veiculos=n, demanda=DemandaPoisson(0), algoritmo=Hotspot(),
                    seed=0, previsao=previsao)

    total = {z: sum(previsao.lam(z, p) for p in range(144)) for z in grade.zonas()}
    fracao = {z: t / sum(total.values()) for z, t in total.items()}
    bases = [v.base for v in sim.veiculos]

    maior = max(fracao, key=fracao.get)  # a célula (5, 3), ~9,2% da demanda
    esperado, desvio = fracao[maior] * n, np.sqrt(n * fracao[maior] * (1 - fracao[maior]))
    assert abs(bases.count(maior) - esperado) < 4 * desvio, (bases.count(maior), esperado)
    zeros = [z for z, t in total.items() if t == 0]  # ex.: a (16, 6), só com desembarques
    assert zeros and not any(b in zeros for b in bases)
