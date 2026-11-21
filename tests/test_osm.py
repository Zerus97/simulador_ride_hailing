import networkx as nx
import pytest

from etl.osm import adicionar_velocidades


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
