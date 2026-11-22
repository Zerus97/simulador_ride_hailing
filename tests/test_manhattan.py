"""1.14: os invariantes no cenário real, montado pelo executor (pula sem o dataset).

Dia 23, frota e passos pequenos; todos os baselines, os dois modelos de aceite e as duas
regras pós-recusa. A cada passo:
  - nenhum veículo criado ou perdido, cada um em exatamente um estado coerente;
  - nenhum pedido criado ou perdido: atendidos + expirados + abertos = total;
e, no fim, as frações de estado somam 1.
"""
from itertools import product
from pathlib import Path

import pandas as pd
import pytest

from environment.env import EM_CORRIDA, OCIOSO, REPOSICIONANDO
from experimentos.cenario import montar, validar

DATASET = Path(__file__).resolve().parent.parent / "dataset"
ARQUIVOS = ("pedidos_1200.parquet", "grade_1200.csv", "nao_vizinhos_1200.csv", "lambda_1200.csv",
            "tempos_1200.csv", "fatores_hora.csv")
FROTA, PASSOS, AQUECIMENTO = 150, 18, 6   # 3h às 6h

# cada baseline com o sem_recomendacao que ele exige
BASELINES = {"sem_movimento": "ficar", "sem_recomendacao": "preferencias",
             "vizinho_aleatorio": "ficar", "gulosa": "ficar"}
COMBINACOES = (list(product(BASELINES, ("fixo", "questionario", "evolutivo"),
                            ("ficar", "proporcional", "preferida", "quatro_preferidas")))
               + [(b, "ashkrof", None) for b in BASELINES]  # o Ashkrof decide o destino da recusa
               + list(product(BASELINES, ("logistica",), ("ficar", "proporcional"))))


@pytest.fixture(scope="module")
def pedidos():
    if not all((DATASET / a).exists() for a in ARQUIVOS):
        pytest.skip("dataset fora do repositório")
    return pd.read_parquet(DATASET / ARQUIVOS[0])


def cenario(algoritmo, modelo, regra):
    comportamento = {"modelo": modelo, "sem_recomendacao": BASELINES[algoritmo]}
    if regra is not None:
        comportamento["regra"] = regra
    if modelo == "fixo":
        comportamento["rho"] = 0.5  # metade recusa: exercita a regra pós-recusa
    if modelo == "evolutivo":
        comportamento["n0"] = 48.0  # "um dia de experiência"
    return validar({
        "cenario": {"nome": f"{algoritmo}-{modelo}-{regra}", "finalidade": "avaliacao", "lado": 1200,
                    "dia": 23, "fracao": 0.15, "fator_demanda": 1.0, "frota": FROTA, "passo_inicial": 18,
                    "passos": PASSOS, "aquecimento": AQUECIMENTO},
        "comportamento": comportamento,
        "algoritmo": {"nome": algoritmo},
        "execucao": {"sementes": [0], "saida": "resultados/teste"},
    })


def conferir_veiculos(sim, ids):
    assert [v.id for v in sim.veiculos] == ids, "veículo criado ou perdido"
    zonas = set(sim.grade.zonas())
    em_corrida = {c.veiculo.id for c in sim.corridas_ativas}
    assert len(em_corrida) == len(sim.corridas_ativas), "veículo em duas corridas"
    for v in sim.veiculos:
        assert v.zona in zonas and v.base in zonas
        assert v.estado in (OCIOSO, EM_CORRIDA, REPOSICIONANDO)
        assert (v.estado == EM_CORRIDA) == (v.id in em_corrida)
        if v.estado == REPOSICIONANDO:
            assert v.destino in sim.grade.vizinhas(v.zona) and v.chegada == sim.tempo + 1
        else:
            assert v.destino is None and v.chegada is None


def conferir_pedidos(sim):
    corridas = sim.corridas_ativas + sim.corridas_concluidas
    atendidos = [c.pedido.id for c in corridas]
    abertos = [p.id for p in sim.pedidos_abertos()]
    assert len(set(atendidos)) == len(atendidos), "pedido atendido duas vezes"
    assert not set(atendidos) & set(abertos), "pedido atendido e ainda aberto"
    assert len(atendidos) + sim.pedidos_expirados + len(abertos) == sim.total_pedidos
    assert [p.id for p in sim.historico_pedidos] == list(range(1, sim.total_pedidos + 1))


@pytest.mark.parametrize("algoritmo, modelo, regra", COMBINACOES,
                         ids=[f"{a}-{m}-{r}" for a, m, r in COMBINACOES])
def test_invariantes_no_dia_23(pedidos, algoritmo, modelo, regra):
    sim = montar(cenario(algoritmo, modelo, regra), pedidos, semente=0)
    ids = [v.id for v in sim.veiculos]
    assert len(ids) == FROTA

    for _ in range(PASSOS):
        sim.passo()
        conferir_veiculos(sim, ids)
        conferir_pedidos(sim)

    m = sim.metricas()
    assert m["frac_parado"] + m["frac_reposicionando"] + m["frac_em_corrida"] == pytest.approx(1)
    assert sum(sim.coletor.estados.values()) == FROTA * (PASSOS - AQUECIMENTO)
    assert 0 <= m["atendimento"] <= 1 and 0 <= m["espera_media_min"] <= 20
    assert sim.total_pedidos > 0
