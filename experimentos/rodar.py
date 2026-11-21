"""Roda um cenário em todas as sementes e grava os resultados.

python -m experimentos.rodar cenarios/base_util.toml

Na pasta de saída do cenário ([execucao] saida):
  - resumo.csv: uma linha por semente, com as métricas de sim.metricas();
  - passos_s{semente}.csv: uma linha por passo (pedidos, casamentos, expirados, carros
    por estado, aceites, recusas, movimentos por tipo, tempo de decisão);
  - execucao.json: o cenário, o commit do git, se havia alterações não commitadas, as
    versões de Python e numpy e a data. É o que permite reproduzir um resultado depois.
"""
import dataclasses
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from environment.env import EM_CORRIDA, OCIOSO, REPOSICIONANDO
from experimentos.cenario import DATASET, ler_cenario, montar
from simulator.metricas import TIPOS_MOVIMENTO

RAIZ = Path(__file__).resolve().parent.parent
COLUNAS_PASSO = ["tempo", "passo_do_dia", "pedidos_novos", "casamentos", "expirados",
                 "ociosos", "em_corrida", "reposicionando", "aceites", "recusas",
                 *(f"mov_{t}" for t in TIPOS_MOVIMENTO), "decisao_s", "confianca_frota"]


def rodar_semente(cenario, pedidos, semente):
    """(sim, tabela por passo) de uma semente. Usa sim.passo() em laço, e não rodar(),
    para montar a linha de cada passo sem mexer no simulador."""
    sim = montar(cenario, pedidos, semente)
    linhas = []
    for _ in range(cenario.passos):
        pedidos_antes, expirados_antes = len(sim.historico_pedidos), sim.pedidos_expirados
        sim.passo()
        rel = sim.ultimo_relatorio
        estados = Counter(v.estado for v in sim.veiculos)
        movimentos = Counter(tipo for *_, tipo in rel.movimentos)
        linhas.append({
            "tempo": sim.tempo,
            "passo_do_dia": sim.fonte_demanda.passo_do_dia(sim.tempo),
            "pedidos_novos": len(sim.historico_pedidos) - pedidos_antes,
            "casamentos": len(rel.casamentos),
            "expirados": sim.pedidos_expirados - expirados_antes,
            "ociosos": estados[OCIOSO], "em_corrida": estados[EM_CORRIDA],
            "reposicionando": estados[REPOSICIONANDO],
            "aceites": len(rel.aceites), "recusas": len(rel.recusas),
            **{f"mov_{t}": movimentos[t] for t in TIPOS_MOVIMENTO},
            "decisao_s": sim.ultimo_tempo_decisao,
            "confianca_frota": sim.comportamento.confianca_media(),
        })
    return sim, pd.DataFrame(linhas, columns=COLUNAS_PASSO)


def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=RAIZ, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None


def registro_da_execucao(cenario):
    status = _git("status", "--porcelain")
    commit = _git("rev-parse", "HEAD")
    return {
        "cenario": dataclasses.asdict(cenario),
        "commit": commit.strip() if commit else None,
        "alteracoes_nao_commitadas": bool(status.strip()) if status is not None else None,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "data": datetime.now().isoformat(timespec="seconds"),
    }


def rodar_cenario(cenario, pedidos, pasta=None):
    """Roda todas as sementes do cenário e grava resumo.csv, passos_s{semente}.csv e
    execucao.json em `pasta` (padrão: [execucao] saida, relativa à raiz do projeto).
    Devolve o resumo (DataFrame)."""
    pasta = Path(pasta) if pasta is not None else RAIZ / cenario.saida
    pasta.mkdir(parents=True, exist_ok=True)
    (pasta / "execucao.json").write_text(
        json.dumps(registro_da_execucao(cenario), indent=2, ensure_ascii=False))

    resumo = []
    for semente in cenario.sementes:
        sim, passos = rodar_semente(cenario, pedidos, semente)
        passos.to_csv(pasta / f"passos_s{semente}.csv", index=False)
        resumo.append({"semente": semente, **sim.metricas()})
        print(f"{cenario.nome}, semente {semente}: atendimento {resumo[-1]['atendimento']:.4f}")
    resumo = pd.DataFrame(resumo)
    resumo.to_csv(pasta / "resumo.csv", index=False)
    return resumo


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("uso: python -m experimentos.rodar cenarios/<cenario>.toml")
    cenario = ler_cenario(sys.argv[1])
    pedidos = pd.read_parquet(DATASET / f"pedidos_{cenario.lado}.parquet")  # lido uma vez
    rodar_cenario(cenario, pedidos)
    print(f"salvo em {RAIZ / cenario.saida}")
