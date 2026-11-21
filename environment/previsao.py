"""Previsão λ: pedidos esperados por zona e passo do dia (proposta §3.2).

Fica no environment/, a camada leve: lê o CSV com o módulo csv, sem pandas.
A previsão já vem na escala dos pedidos simulados (× fração × fator de demanda),
para que nenhum algoritmo precise lembrar de multiplicar.
"""
import csv

from environment.tempo import PASSOS_POR_DIA


class PrevisaoDemanda:
    """λ por (zona, passo do dia), já na escala dos pedidos simulados."""

    def __init__(self, lam_por_zona_passo):
        self._lam = dict(lam_por_zona_passo)

    def lam(self, zona, passo_do_dia):
        """Pedidos esperados com origem em `zona` no passo do dia.

        Passo 144 ou mais é o mesmo passo do dia seguinte do mesmo tipo (resto da
        divisão por 144); só importa na última decisão, ao olhar um passo à frente.
        Acesso direto, sem .get(..., 0): a tabela é densa, então uma zona
        desconhecida é erro e aparece como KeyError."""
        return self._lam[(zona, passo_do_dia % PASSOS_POR_DIA)]


class PrevisaoConstante:
    """Para os testes com Poisson: o mesmo λ em toda zona e todo passo, que é
    exatamente a demanda que a DemandaPoisson gera."""

    def __init__(self, lam):
        self.valor = lam

    def lam(self, zona, passo_do_dia):
        return self.valor


def previsao_do_csv(caminho, grade, tipo_dia, fracao, fator=1.0):
    """PrevisaoDemanda a partir do lambda_{lado}.csv (etl/demanda.py), para um tipo de dia
    ("util" ou "sabado"), já multiplicada por fração × fator de demanda.

    Confere que a tabela é densa: toda zona da grade em todo passo do dia."""
    lam = {}
    with open(caminho, newline="") as f:
        for r in csv.DictReader(f):
            if r["tipo_dia"] != tipo_dia:
                continue
            zona = grade.zona((int(r["linha"]), int(r["coluna"])))
            lam[(zona, int(r["passo"]))] = float(r["pedidos_medios"]) * fracao * fator

    faltam = [(z, p) for z in grade.zonas() for p in range(PASSOS_POR_DIA) if (z, p) not in lam]
    if faltam:
        raise ValueError(f"previsão de {tipo_dia!r} sem {len(faltam)} (zona, passo), ex.: {faltam[:3]}")
    return PrevisaoDemanda(lam)
