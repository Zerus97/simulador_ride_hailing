import networkx as nx
import pandas as pd
import pytest

from etl.grade import para_latlon
from etl.osm import adicionar_velocidades, nao_vizinhos, pontos_mediana


def test_velocidade_pelo_tipo_ignora_maxspeed():
    # o OSMnx precisa do sistema de coordenadas e de x/y nos nós para calcular os tempos
    g = nx.MultiDiGraph(crs="EPSG:4326")
    g.add_node(1, x=-73.98, y=40.75)
    g.add_node(2, x=-73.97, y=40.76)
    g.add_edge(1, 2, highway="motorway", length=1000.0, maxspeed="120 mph")  # erro do mapa de 2016
    g.add_edge(2, 1, highway=["residential", "secondary"], length=1000.0, maxspeed="30")

    g = adicionar_velocidades(g)

    expressa, rua = g[1][2][0], g[2][1][0]
    assert expressa["speed_kph"] == 64.4 and expressa["travel_time"] == pytest.approx(55.90, abs=0.01)
    assert rua["speed_kph"] == 40.2 and rua["travel_time"] == pytest.approx(89.55, abs=0.01)


# --- pontos_mediana ---
# Células de 1000 m com origem em (583000, 4506000), em metros (UTM 18N):
#   (0, 0): x de 583000 a 584000 | (0, 1): x de 584000 a 585000 | (1, 0): sem viagens
ORIGEM = (583_000, 4_506_000)
LADO = 1000


def viagens_de_mentira(embarques, desembarques):
    """DataFrame de viagens a partir de pontos conhecidos em metros."""
    (xe, ye), (xd, yd) = zip(*embarques), zip(*desembarques)
    lon_e, lat_e = para_latlon.transform(list(xe), list(ye))
    lon_d, lat_d = para_latlon.transform(list(xd), list(yd))
    return pd.DataFrame({"pickup_longitude": lon_e, "pickup_latitude": lat_e,
                         "dropoff_longitude": lon_d, "dropoff_latitude": lat_d})


# três viagens: embarques em (0, 0), desembarques em (0, 1)
VIAGENS = viagens_de_mentira(
    embarques=[(583_100, 4_506_100), (583_300, 4_506_300), (583_900, 4_506_200)],
    desembarques=[(584_500, 4_506_500), (584_700, 4_506_900), (584_600, 4_506_700)],
)


def test_pontos_mediana_da_o_valor_calculado_a_mao():
    # a grade em ordem diferente da contagem: o resultado segue a ordem da grade
    grade = pd.DataFrame({"linha": [0, 0], "coluna": [1, 0]})

    x, y = pontos_mediana(VIAGENS, grade, LADO, ORIGEM)

    # (0, 1): x = mediana(584500, 584700, 584600) = 584600; y = mediana(4506500, 4506900, 4506700)
    # (0, 0): x = mediana(583100, 583300, 583900) = 583300; y = mediana(4506100, 4506300, 4506200)
    assert x == pytest.approx([584_600, 583_300], abs=1e-3)
    assert y == pytest.approx([4_506_700, 4_506_200], abs=1e-3)


def test_pontos_mediana_celula_sem_viagens_da_erro():
    grade = pd.DataFrame({"linha": [0, 0, 1], "coluna": [0, 1, 0]})
    with pytest.raises(KeyError, match=r"\(1, 0\)"):
        pontos_mediana(VIAGENS, grade, LADO, ORIGEM)


# --- nao_vizinhos (1.3d) ---

def test_nao_vizinhos_um_sentido_fronteira_e_sem_repeticao():
    # A = (0, 0), B = (0, 1), C = (1, 0); tempos pelo OSM nos dois sentidos
    linhas = [((0, 0), (0, 1), 3.0), ((0, 1), (0, 0), 4.0),    # A-B: 4,0 x 4 = 16,0 -> sai (basta um sentido)
              ((0, 0), (1, 0), 3.75), ((1, 0), (0, 0), 3.0),   # A-C: 3,75 x 4 = 15,0 -> sai (a regra é >= 15)
              ((0, 1), (1, 0), 3.0), ((1, 0), (0, 1), 3.7)]    # B-C: 3,7 x 4 = 14,8 -> fica
    tempos = pd.DataFrame([{"linha_a": a[0], "coluna_a": a[1], "linha_b": b[0], "coluna_b": b[1],
                            "osm_min": m} for a, b, m in linhas])

    excluidos = nao_vizinhos(tempos, maior_fator=4.0)

    pares = list(zip(excluidos["linha_a"], excluidos["coluna_a"], excluidos["linha_b"], excluidos["coluna_b"]))
    assert pares == [(0, 0, 0, 1), (0, 0, 1, 0)]
    assert excluidos["calibrado_pior_min"].tolist() == pytest.approx([16.0, 15.0])
