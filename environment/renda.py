"""Renda esperada por zona (para a logística do i-Rebalance, tarefa 1.10d).

Fica no environment/, a camada leve: lê o CSV com o módulo csv, sem pandas.
"""
import csv

TARIFA_USD_KM = 3.25  # Q4: soma das tarifas ÷ soma dos km nos dias de treino (fonte única)


class RendaEsperada:
    """O fator de renda de cada zona: tarifa média das corridas que começam nela ÷ a
    média geral (ponderada pelo número de corridas). Zona sem corridas: 1,0."""

    def __init__(self, fator_por_zona):
        self._fator = dict(fator_por_zona)

    def fator(self, zona):
        return self._fator[zona]


def renda_do_csv(caminho, grade, tipo_dia):
    """RendaEsperada a partir do renda_{lado}.csv (etl/demanda.py), para um tipo de dia."""
    corridas, tarifa = {}, {}
    with open(caminho, newline="") as f:
        for r in csv.DictReader(f):
            if r["tipo_dia"] != tipo_dia:
                continue
            zona = grade.zona((int(r["linha"]), int(r["coluna"])))
            corridas[zona] = int(r["corridas"])
            tarifa[zona] = float(r["tarifa_media"]) if corridas[zona] else 0.0
    faltam = [z for z in grade.zonas() if z not in corridas]
    if faltam:
        raise ValueError(f"renda de {tipo_dia!r} sem {len(faltam)} zonas da grade, ex.: {faltam[:3]}")
    total = sum(corridas.values())
    media = sum(corridas[z] * tarifa[z] for z in corridas) / total
    return RendaEsperada({z: (tarifa[z] / media if corridas[z] else 1.0) for z in corridas})
