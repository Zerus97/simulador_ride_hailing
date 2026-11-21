"""Tempos e distâncias entre zonas vizinhas (tarefa 1.3d), para os algoritmos.

Fica no environment/, a camada leve: lê os CSVs com o módulo csv, sem pandas.
O tempo calibrado é o tempo pelo OSM × fator da hora (o da faixa das vizinhas,
fatores_hora.csv); a distância é a do OSM, sem fator.
"""
import csv

HORAS_POR_DIA = 24


class TemposEntreZonas:
    """Minutos (calibrados por hora) e km entre zonas vizinhas.

    Ficar (origem igual ao destino) dá 0 minutos e 0 km. Par que não é vizinho dá
    KeyError."""

    def __init__(self, osm_min, km, fator_por_hora):
        self._osm_min = dict(osm_min)        # (origem, destino) -> minutos pelo OSM
        self._km = dict(km)                  # (origem, destino) -> km do caminho mais rápido
        self._fator = dict(fator_por_hora)   # hora (0 a 23) -> fator

    def minutos(self, origem, destino, hora):
        """osm_min × fator[hora]. A hora de um passo do dia é passo_do_dia // 6; hora 24
        ou mais (a última decisão do dia, olhando à frente) usa a do dia seguinte."""
        if origem == destino:
            return 0.0
        return self._osm_min[(origem, destino)] * self._fator[hora % HORAS_POR_DIA]

    def km(self, origem, destino):
        if origem == destino:
            return 0.0
        return self._km[(origem, destino)]

    def pares(self):
        """Os pares ordenados (origem, destino) com tempo."""
        return list(self._osm_min)


def tempos_do_csv(caminho_tempos, caminho_fatores, grade, lado, tipo_dia):
    """TemposEntreZonas a partir do tempos_{lado}.csv e do fatores_hora.csv (coluna fator,
    o da faixa das vizinhas), para a grade (já sem os não vizinhos) e um tipo de dia.

    Guarda só os pares (zona, vizinha) da grade e confere que toda vizinha tem tempo:
    isso valida que os arquivos combinam com a grade."""
    vizinhos = {(z, v) for z in grade.zonas() for v in grade.vizinhas(z)}
    osm_min, km = {}, {}
    with open(caminho_tempos, newline="") as f:
        for r in csv.DictReader(f):
            a, b = (int(r["linha_a"]), int(r["coluna_a"])), (int(r["linha_b"]), int(r["coluna_b"]))
            try:
                par = (grade.zona(a), grade.zona(b))
            except KeyError as erro:
                raise ValueError(f"o arquivo de tempos tem a célula {erro}, que não está na grade") from None
            if par in vizinhos:
                osm_min[par] = float(r["osm_min"])
                km[par] = float(r["km"])

    faltam = sorted(vizinhos - set(osm_min))
    if faltam:
        raise ValueError(f"{len(faltam)} pares de vizinhas sem tempo, ex.: "
                         f"{[(grade.celula(a), grade.celula(b)) for a, b in faltam[:3]]}")

    fator = {}
    with open(caminho_fatores, newline="") as f:
        for r in csv.DictReader(f):
            if int(r["lado"]) == lado and r["tipo_dia"] == tipo_dia:
                fator[int(r["hora"])] = float(r["fator"])
    if set(fator) != set(range(HORAS_POR_DIA)):
        raise ValueError(f"fatores de {lado} m, {tipo_dia!r}: horas {sorted(fator)}, esperado 0 a 23")
    return TemposEntreZonas(osm_min, km, fator)
