"""Calibração da frota (tarefa 1.15): a frota que dá ~75% de atendimento sem reposicionamento.

python -m experimentos.calibrar_frota

Escolhida nos DIAS ÚTEIS DE TREINO (finalidade "ajuste"), e não no dia de avaliação,
para não ajustar o cenário com o dia que vai ser avaliado. A regra foi fixada antes de
ver a curva: a frota da grade (de 100 em 100) cujo atendimento médio, na média dos 7 dias
de treino, fique mais perto de 75%, o meio da faixa de 70–80% da proposta; o meio dá
margem para os dias de avaliação, que variam um pouco.

Só depois da escolha, o dia 23 (útil) e o sábado 19 rodam com a frota escolhida, para
registrar onde caíram. Esses números não voltam para a escolha.

Grava em resultados/calibracao_frota/: por_dia.csv (dia × frota × semente), curva.csv
(média e desvio entre dias, por frota), avaliacao.csv e escolha.json.
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd

from environment.calendario import DIAS_AVALIACAO, DIAS_TREINO
from experimentos.cenario import DATASET, montar, validar
from experimentos.rodar import registro_da_execucao

RAIZ = Path(__file__).resolve().parent.parent
SAIDA = RAIZ / "resultados" / "calibracao_frota"
FROTAS = list(range(1000, 2501, 100))
SEMENTES_CURVA = [0, 1, 2]
SEMENTES_AVALIACAO = list(range(10))
ALVO = 0.75
LADO, FRACAO, PASSO_INICIAL, PASSOS, AQUECIMENTO = 1200, 0.15, 18, 126, 6


def cenario(dia, frota, finalidade, sementes):
    return validar({
        "cenario": {"nome": f"frota_{frota}_dia_{dia}", "finalidade": finalidade, "lado": LADO, "dia": dia,
                    "fracao": FRACAO, "fator_demanda": 1.0, "frota": frota, "passo_inicial": PASSO_INICIAL,
                    "passos": PASSOS, "aquecimento": AQUECIMENTO},
        "comportamento": {"modelo": "fixo", "rho": 1.0, "regra": "ficar", "sem_recomendacao": "ficar"},
        "algoritmo": {"nome": "sem_movimento"},
        "execucao": {"sementes": sementes, "saida": "resultados/calibracao_frota"},
    })


def atendimento(c, pedidos, semente):
    sim = montar(c, pedidos, semente)
    sim.rodar(c.passos)
    return sim.metricas()["atendimento"]


def rodar_dias(dias, frotas, sementes, finalidade, pedidos):
    linhas = []
    for dia in dias:
        for frota in frotas:
            c = cenario(dia, frota, finalidade, sementes)
            for s in sementes:
                linhas.append({"dia": dia, "frota": frota, "semente": s, "atendimento": atendimento(c, pedidos, s)})
            print(f"dia {dia:>2}, frota {frota}: "
                  f"{pd.DataFrame(linhas[-len(sementes):])['atendimento'].mean():.4f}", flush=True)
    return pd.DataFrame(linhas)


def curva(por_dia):
    """Por frota: média dos 7 dias (cada dia = média das sementes) e desvio entre os dias."""
    medias_do_dia = por_dia.groupby(["frota", "dia"])["atendimento"].mean()
    return medias_do_dia.groupby("frota").agg(atendimento_medio="mean", desvio_entre_dias="std",
                                              minimo_dia="min", maximo_dia="max").reset_index()


def escolher(tabela, alvo=ALVO):
    """A frota cujo atendimento médio fica mais perto do alvo (empate: a menor)."""
    distancia = (tabela["atendimento_medio"] - alvo).abs()
    return int(tabela.loc[distancia.idxmin(), "frota"])


if __name__ == "__main__":
    inicio = time.time()
    SAIDA.mkdir(parents=True, exist_ok=True)
    pedidos = pd.read_parquet(DATASET / f"pedidos_{LADO}.parquet")

    por_dia = rodar_dias(DIAS_TREINO, FROTAS, SEMENTES_CURVA, "ajuste", pedidos)
    por_dia.to_csv(SAIDA / "por_dia.csv", index=False)
    tabela = curva(por_dia)
    tabela.to_csv(SAIDA / "curva.csv", index=False)
    frota = escolher(tabela)
    print(tabela.round(4).to_string(index=False))
    print(f"\nfrota escolhida (mais perto de {ALVO:.0%} nos dias de treino): {frota}")

    # só depois da escolha: onde os dias de avaliação caem (não volta para a escolha)
    avaliacao = rodar_dias(sorted(DIAS_AVALIACAO), [frota], SEMENTES_AVALIACAO, "avaliacao", pedidos)
    avaliacao.to_csv(SAIDA / "avaliacao.csv", index=False)
    resumo = avaliacao.groupby("dia")["atendimento"].agg(["mean", "std", "min", "max"])
    print(resumo.round(4).to_string())

    escolhida = tabela[tabela["frota"] == frota].iloc[0]
    (SAIDA / "escolha.json").write_text(json.dumps({
        "regra": f"frota da grade {FROTAS[0]}–{FROTAS[-1]} (passo 100) mais perto de {ALVO} "
                 f"no atendimento médio dos dias úteis de treino {DIAS_TREINO}",
        "frota": frota,
        "treino": {"atendimento_medio": escolhida["atendimento_medio"],
                   "desvio_entre_dias": escolhida["desvio_entre_dias"]},
        "avaliacao": {int(d): {"media": r["mean"], "desvio": r["std"], "sementes": len(SEMENTES_AVALIACAO)}
                      for d, r in resumo.iterrows()},
        "execucao": registro_da_execucao(cenario(23, frota, "avaliacao", SEMENTES_AVALIACAO)),
        "duracao_s": round(time.time() - inicio),
    }, indent=2, ensure_ascii=False))
    print(f"salvo em {SAIDA} ({time.time() - inicio:.0f} s)", file=sys.stderr)
