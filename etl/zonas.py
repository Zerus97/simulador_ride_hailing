"""Zonas de táxi de Nova York (shapefile da TLC).

Os dados ficam em dataset/taxi_zones/ (fora do git, pelo .gitignore).
"""
from functools import cache
from pathlib import Path

import geopandas as gpd
import shapely

ARQUIVO_ZONAS = Path(__file__).resolve().parent.parent / "dataset" / "taxi_zones" / "taxi_zones.shp"

# Ilhas sem acesso por carro: no OpenStreetMap não teriam rota para o resto
# da cidade, e o cálculo dos tempos de viagem (tarefa 1.3) quebraria.
EXCLUIDAS = ["Governor's Island/Ellis Island/Liberty Island"]


def ler_zonas(caminho=ARQUIVO_ZONAS):
    """Todas as zonas de táxi, com os polígonos na projeção do arquivo (EPSG:2263, pés)."""
    return gpd.read_file(caminho)


def zonas_manhattan(zonas=None, excluir=EXCLUIDAS):
    """Zonas de Manhattan, sem as zonas em `excluir`."""
    if zonas is None:
        zonas = ler_zonas()
    manhattan = zonas[zonas["borough"] == "Manhattan"]
    return manhattan[~manhattan["zone"].isin(excluir)]


def contorno_latlon(zonas):
    """Une as zonas num único contorno em lat/lon (EPSG:4326), preparado
    para testes repetidos de ponto dentro/fora."""
    poligono = zonas.to_crs(4326).union_all()
    shapely.prepare(poligono)
    return poligono


@cache
def poligono_manhattan():
    """Contorno de Manhattan (sem as ilhas excluídas), em lat/lon."""
    return contorno_latlon(zonas_manhattan())


# Q5: a grade é montada em UTM zona 18N (metros), com células alinhadas ao norte
# e origem no canto inferior esquerdo do contorno de Manhattan.
CRS_GRADE = 32618


@cache
def contorno_metros():
    """Contorno de Manhattan (sem as ilhas excluídas) em UTM 18N, em metros."""
    return zonas_manhattan().to_crs(CRS_GRADE).union_all()


def origem_grade():
    """(x, y) do canto inferior esquerdo do contorno, em metros (UTM 18N)."""
    minx, miny, _, _ = contorno_metros().bounds
    return minx, miny


def dentro(poligono, lon, lat):
    """True/False para cada ponto. Atenção: longitude (x) antes de latitude (y)."""
    return shapely.contains_xy(poligono, lon, lat)


if __name__ == "__main__":
    zonas = ler_zonas()

    print("== colunas ==")
    print(list(zonas.columns))
    faltando = {"LocationID", "zone", "borough"} - set(zonas.columns)
    print(f"faltando: {faltando or 'nenhuma'}")

    print("\n== projeção ==")
    print(zonas.crs)
    print(f"EPSG: {zonas.crs.to_epsg()} | unidade: {zonas.crs.axis_info[0].unit_name}")

    todas = zonas_manhattan(zonas, excluir=[])
    manhattan = zonas_manhattan(zonas)
    print(f"\n== zonas de Manhattan ({len(manhattan)} de {len(todas)}; "
          f"excluídas: {sorted(set(todas['LocationID']) - set(manhattan['LocationID']))}) ==")
    print(manhattan[["LocationID", "zone"]].to_string(index=False))

    minx, miny, maxx, maxy = contorno_metros().bounds
    print(f"\n== grade (EPSG:{CRS_GRADE}, metros, alinhada ao norte) ==")
    print(f"origem (canto inferior esquerdo): x = {minx:,.1f}  y = {miny:,.1f}")
    print(f"extensão: {maxx - minx:,.0f} m (leste-oeste) x {maxy - miny:,.0f} m (norte-sul)")
