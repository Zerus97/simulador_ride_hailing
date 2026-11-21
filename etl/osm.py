"""
Rede viária de Manhattan (OpenStreetMap) e conferência das velocidades.

python -m etl.osm --rede   1.3a: baixa a rede na data fixa (se ainda não houver),
                           confere e mostra os metadados
python -m etl.osm          compara, por hora do dia, a duração registrada das viagens
                           de treino com o tempo do caminho mínimo no OSM, com as
                           velocidades padrão do OSMnx, e calcula os tempos entre vizinhas
"""
import json
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from etl.etl import DATASET, DIAS_TREINO, SAIDA, duracao_min
from etl.grade import celula, para_latlon, para_metros, pontas_em_metros
from etl.zonas import poligono_manhattan

REDE = DATASET / "manhattan_drive.graphml"
METADADOS = DATASET / "manhattan_drive.json"
# Rede "como estava" no início do mês das viagens. O Overpass responde com os dados
# daquele instante, então quem baixar de novo recebe a mesma rede.
DATA_REDE = "2016-03-01T00:00:00Z"
TIPO_REDE = "drive"
MARGEM_GRAUS = 0.001  # ~100 m além do contorno, para não cortar as vias da margem (FDR, West Side)
AMOSTRA = 200_000
SEMENTE = 0

LIMITE_RUA_KMH = 40.2       # 25 mph: ruas sem placa em NYC desde 7/11/2014
LIMITE_EXPRESSA_KMH = 64.4  # 40 mph: limite da FDR Drive
VIAS_EXPRESSAS = {"motorway", "trunk"}  # os ramais (_link) ficam com o limite das ruas


def tipo_via(highway):
    """O OSMnx guarda uma lista quando junta trechos de tipos diferentes; vale o primeiro."""
    return highway[0] if isinstance(highway, list) else highway


def adicionar_velocidades(g):
    """Velocidade de cada trecho pelo tipo de via, com os limites de 2016.

    Ignora o maxspeed do mapa de 2016, que é raro (190 de 9.633 trechos) e tem
    erros, como 120 mph na FDR (ver arquivo de decisões, seção 2)."""
    for _, _, dados in g.edges(data=True):
        expressa = tipo_via(dados["highway"]) in VIAS_EXPRESSAS
        dados["speed_kph"] = LIMITE_EXPRESSA_KMH if expressa else LIMITE_RUA_KMH
    return ox.add_edge_travel_times(g)


def _timestamp_overpass(desde):
    """timestamp_osm_base das respostas do Overpass gravadas no cache depois de `desde`."""
    pasta = Path(ox.settings.cache_folder)
    stamps = set()
    for arquivo in pasta.glob("*.json"):
        if arquivo.stat().st_mtime >= desde:
            with open(arquivo) as f:
                inicio = f.read(500)
            if '"osm3s"' in inicio:
                stamps.add(inicio.split('"timestamp_osm_base": "')[1].split('"')[0])
    return sorted(stamps)


def baixar_rede():
    """1.3a: baixa a rede bruta na DATA_REDE e salva graphml + JSON de metadados.

    A rede é salva como veio do OSMnx (sem velocidades e sem o corte no maior
    componente): os dois são aplicados ao carregar, em preparar_rede(). Assim,
    mudar uma velocidade não exige baixar a rede de novo."""
    ox.settings.overpass_settings = f'[out:json][timeout:{{timeout}}]{{maxsize}}[date:"{DATA_REDE}"]'
    inicio = datetime.now(timezone.utc)
    bruta = ox.graph_from_polygon(poligono_manhattan().buffer(MARGEM_GRAUS), network_type=TIPO_REDE)
    ox.save_graphml(bruta, REDE)

    # Se a resposta veio do cache do OSMnx (mesma consulta já feita), não há arquivo
    # novo no cache: mantém a data da base e do download da primeira vez.
    stamps = _timestamp_overpass(inicio.timestamp())
    anterior = metadados()
    do_cache = not stamps and anterior is not None and anterior.get("data_rede") == DATA_REDE
    componente = ox.truncate.largest_component(bruta, strongly=True)
    meta = {
        "data_rede": DATA_REDE,
        "overpass_url": ox.settings.overpass_url,
        "overpass_settings": ox.settings.overpass_settings,
        "timestamp_osm_base": anterior["timestamp_osm_base"] if do_cache else stamps,
        "baixado_em": anterior["baixado_em"] if do_cache else inicio.isoformat(timespec="seconds"),
        "tipo_rede": TIPO_REDE,
        "margem_graus": MARGEM_GRAUS,
        "versoes": {p: version(p) for p in ("osmnx", "networkx", "shapely", "geopandas")},
        "nos_baixados": bruta.number_of_nodes(),
        "arestas_baixadas": bruta.number_of_edges(),
        "nos_maior_componente": componente.number_of_nodes(),
        "arestas_maior_componente": componente.number_of_edges(),
    }
    METADADOS.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return bruta


def preparar_rede(bruta):
    """Velocidades pelo tipo de via e só o maior componente fortemente conexo.

    O maior componente garante que de qualquer nó se chega a qualquer outro. Sem
    isso, um ponto ligado a um beco de mão única não tem caminho para as vizinhas
    (tempo infinito)."""
    g = adicionar_velocidades(bruta)
    g = ox.truncate.largest_component(g, strongly=True)
    conferir_rede(g)
    return g


def conferir_rede(g):
    """A rede precisa ser fortemente conexa, ter tempo de viagem positivo em toda
    aresta e só as velocidades da regra."""
    assert g.number_of_nodes() > 0, "rede vazia"
    assert nx.is_strongly_connected(g), "rede não é fortemente conexa"
    tempos = [d.get("travel_time") for _, _, d in g.edges(data=True)]
    assert all(t is not None and t > 0 for t in tempos), "aresta sem travel_time positivo"
    velocidades = {d.get("speed_kph") for _, _, d in g.edges(data=True)}
    assert velocidades <= {LIMITE_RUA_KMH, LIMITE_EXPRESSA_KMH}, f"velocidade fora da regra: {velocidades}"


def metadados():
    return json.loads(METADADOS.read_text()) if METADADOS.exists() else None


def rede():
    """Rede de ruas para carro na DATA_REDE, pronta para uso (velocidades, tempos
    e maior componente). Baixa a rede bruta se não houver uma salva dessa data."""
    meta = metadados()
    if REDE.exists() and meta and meta["data_rede"] == DATA_REDE:
        bruta = ox.load_graphml(REDE)
    else:
        bruta = baixar_rede()
    return preparar_rede(bruta)


def matriz_tempos(g):
    """Grafo -> (matriz esparsa de tempos em s, ids dos nós). Entre arestas
    paralelas fica a mais rápida."""
    dg = ox.convert.to_digraph(g, weight="travel_time")
    nos = list(dg.nodes)
    return nx.to_scipy_sparse_array(dg, nodelist=nos, weight="travel_time"), nos


def no_mais_proximo_xy(g, nos, x, y):
    """Índice (em `nos`) do nó mais próximo de cada ponto (x, y) em metros (UTM 18N)."""
    xs, ys = para_metros.transform([g.nodes[n]["x"] for n in nos], [g.nodes[n]["y"] for n in nos])
    arvore = cKDTree(np.column_stack([xs, ys]))
    distancia, indice = arvore.query(np.column_stack([x, y]))
    return indice, distancia


def no_mais_proximo(g, nos, lon, lat):
    """Índice (em `nos`) do nó mais próximo de cada ponto em lat/lon, medindo em metros."""
    return no_mais_proximo_xy(g, nos, *para_metros.transform(lon, lat))


def tempos_osm(matriz, origem, destino):
    """Tempo do caminho mínimo (s) de cada origem ao seu destino."""
    tempos = np.empty(len(origem))
    unicas, posicao = np.unique(origem, return_inverse=True)
    for bloco in range(0, len(unicas), 500):  # 500 origens por vez para limitar a memória
        fontes = unicas[bloco:bloco + 500]
        dist = dijkstra(matriz, indices=fontes)
        sel = (posicao >= bloco) & (posicao < bloco + len(fontes))
        tempos[sel] = dist[posicao[sel] - bloco, destino[sel]]
    return tempos


def conferir_velocidades(df, g, imprimir=True):
    treino = df[df["tpep_pickup_datetime"].dt.day.isin(DIAS_TREINO)]
    amostra = treino.sample(min(AMOSTRA, len(treino)), random_state=SEMENTE)

    matriz, nos = matriz_tempos(g)
    origem, d_o = no_mais_proximo(g, nos, amostra["pickup_longitude"].values, amostra["pickup_latitude"].values)
    destino, d_d = no_mais_proximo(g, nos, amostra["dropoff_longitude"].values, amostra["dropoff_latitude"].values)

    res = pd.DataFrame({
        "hora": amostra["tpep_pickup_datetime"].dt.hour.values,
        "registrado_min": duracao_min(amostra).values,
        "osm_min": tempos_osm(matriz, origem, destino) / 60,
    })
    validas = (origem != destino) & np.isfinite(res["osm_min"]) & (res["osm_min"] > 0)
    res = res[validas]

    por_hora = res.groupby("hora").agg(
        viagens=("osm_min", "size"),
        registrado_min=("registrado_min", "median"),
        osm_min=("osm_min", "median"),
    )
    soma = res.groupby("hora")[["registrado_min", "osm_min"]].sum()
    por_hora["razao_somas"] = soma["registrado_min"] / soma["osm_min"]
    por_hora["razao_mediana"] = (res["registrado_min"] / res["osm_min"]).groupby(res["hora"]).median()
    if imprimir:
        print(f"rede: {g.number_of_nodes():,} nós, {g.number_of_edges():,} arestas")
        print(f"amostra de treino: {len(validas):,} viagens | distância mediana ao nó: "
              f"{np.median(np.r_[d_o, d_d]):.0f} m")
        print(f"descartadas (mesmo nó ou sem caminho): {(~validas).sum():,}")
        print("\nregistrado ÷ OSM, por hora do embarque (min = medianas):")
        print(por_hora.round(2).to_string())
        total = res["registrado_min"].sum() / res["osm_min"].sum()
        print(f"\ngeral: razão das somas = {total:.2f} | "
              f"mediana das razões = {(res['registrado_min'] / res['osm_min']).median():.2f}")
    return por_hora


# Mesma vizinhança da GradeQuadrada (environment/env.py): 4 lados + 4 diagonais.
VIZINHANCA = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
PASSO_MIN = 10
LIMITE_2_PASSOS = 1.5 * PASSO_MIN  # acima de 15 min, round(t / 10) dá 2 passos


def pares_vizinhos(grade):
    """Pares ordenados (i, j) de índices de `grade` que são vizinhos (a->b e b->a)."""
    indice = {(l, c): i for i, (l, c) in enumerate(zip(grade["linha"], grade["coluna"]))}
    return [(i, indice[(l + dl, c + dc)])
            for (l, c), i in indice.items()
            for dl, dc in VIZINHANCA if (l + dl, c + dc) in indice]


def pontos_mediana(df, grade, lado, origem):
    """(x, y) em metros de cada célula: mediana dos embarques e desembarques nela.

    Uma célula da grade sem nenhuma viagem em `df` dá KeyError com a lista delas."""
    (xe, ye), (xd, yd) = pontas_em_metros(df)
    x, y = np.r_[xe, xd], np.r_[ye, yd]
    linha, coluna = celula(x, y, lado, origem)
    med = pd.DataFrame({"linha": linha, "coluna": coluna, "x": x, "y": y}) \
        .groupby(["linha", "coluna"])[["x", "y"]].median()
    celulas = list(zip(grade["linha"], grade["coluna"]))
    sem_viagens = [c for c in celulas if c not in med.index]
    if sem_viagens:
        raise KeyError(f"células sem viagens nestes dados: {sem_viagens}")
    med = med.loc[celulas]
    return med["x"].values, med["y"].values


LIMITE_DIST_NO_M = 200  # ponto mais longe que isso de qualquer cruzamento: água ou parque


def pontos_zonas(df, grade, lado, origem, g):
    """1.3b: um ponto por célula (mediana das viagens de TREINO) ligado ao cruzamento
    mais próximo da rede. Salva dataset/pontos_{lado}.csv e devolve a tabela.

    Só os dias de treino: nada do dia de avaliação entra na construção do modelo.
    O cruzamento vai pelo id do OSM, que não muda se a ordem da lista de nós mudar."""
    treino = df[df["tpep_pickup_datetime"].dt.day.isin(DIAS_TREINO)]
    x, y = pontos_mediana(treino, grade, lado, origem)
    nos = list(g.nodes)
    indice, distancia = no_mais_proximo_xy(g, nos, x, y)
    lon, lat = para_latlon.transform(x, y)
    pontos = pd.DataFrame({
        "linha": grade["linha"].values, "coluna": grade["coluna"].values,
        "x": x, "y": y, "lon": lon, "lat": lat,
        "no_osm": [nos[i] for i in indice],
        "dist_no_m": distancia,
    })
    pontos.to_csv(DATASET / f"pontos_{lado}.csv", index=False)
    return pontos


def conferir_pontos(pontos, lado):
    d = pontos["dist_no_m"]
    print(f"\n== pontos {lado} m: {len(pontos)} células ==")
    print(f"distância ao cruzamento: mediana {d.median():.0f} m | máximo {d.max():.0f} m")
    longe = pontos[d > LIMITE_DIST_NO_M]
    print(f"acima de {LIMITE_DIST_NO_M} m: {len(longe)}")
    if len(longe):
        print(longe[["linha", "coluna", "lat", "lon", "no_osm", "dist_no_m"]]
              .round({"lat": 5, "lon": 5, "dist_no_m": 0}).to_string(index=False))


def tempos_vizinhos(g, grade, x, y, fatores):
    """Tabela longa: um par de vizinhos por hora, com o tempo do OSM e o calibrado."""
    matriz, nos = matriz_tempos(g)
    no_celula, _ = no_mais_proximo_xy(g, nos, x, y)
    dist = dijkstra(matriz, indices=no_celula)  # células x nós, em segundos
    pares = pares_vizinhos(grade)
    a = np.array([i for i, _ in pares])
    b = np.array([j for _, j in pares])
    pares_df = pd.DataFrame({
        "origem_linha": grade["linha"].values[a], "origem_coluna": grade["coluna"].values[a],
        "destino_linha": grade["linha"].values[b], "destino_coluna": grade["coluna"].values[b],
        "osm_min": dist[a, no_celula[b]] / 60,
        "mesmo_no": no_celula[a] == no_celula[b],
    })
    horas = pd.DataFrame({"hora": fatores.index, "fator": fatores.values})
    tabela = pares_df.merge(horas, how="cross")
    tabela["calibrado_min"] = tabela["osm_min"] * tabela["fator"]
    tabela["passa_15"] = tabela["calibrado_min"] > LIMITE_2_PASSOS
    return tabela


def resumir(tabela, nome):
    pares = tabela.drop_duplicates(["origem_linha", "origem_coluna", "destino_linha", "destino_coluna"])
    print(f"\n== {nome} ==")
    print(f"pares ordenados de vizinhos: {len(pares)} | no mesmo nó (tempo 0): {pares['mesmo_no'].sum()}")
    print(f"OSM (min): mediana {pares['osm_min'].median():.2f} | máx {pares['osm_min'].max():.2f}")
    por_hora = tabela.groupby("hora").agg(
        fator=("fator", "first"),
        calibrado_mediana=("calibrado_min", "median"),
        calibrado_max=("calibrado_min", "max"),
        frac_passa_15=("passa_15", "mean"),
    )
    print(por_hora.round(3).to_string())
    print(f"fração (par, hora) acima de {LIMITE_2_PASSOS:.0f} min: {tabela['passa_15'].mean():.2%}")


if __name__ == "__main__":
    import argparse

    from etl.grade import LADOS
    from etl.zonas import origem_grade

    parser = argparse.ArgumentParser(description="Rede viária de Manhattan (OpenStreetMap).")
    parser.add_argument("--rede", action="store_true",
                        help="1.3a: só baixa (se preciso) e confere a rede, e mostra os metadados")
    parser.add_argument("--pontos", action="store_true",
                        help="1.3b: ponto de cada célula ligado ao cruzamento mais próximo")
    args = parser.parse_args()

    if args.rede:
        g = rede()
        print(json.dumps(metadados(), indent=2, ensure_ascii=False))
        raise SystemExit

    if args.pontos:
        df = pd.read_parquet(SAIDA)
        g = rede()
        origem = origem_grade()
        for lado in LADOS:
            grade = pd.read_csv(DATASET / f"grade_{lado}.csv")
            conferir_pontos(pontos_zonas(df, grade, lado, origem, g), lado)
        raise SystemExit

    df = pd.read_parquet(SAIDA)
    g = rede()
    fatores = conferir_velocidades(df, g)["razao_somas"]
    origem = origem_grade()

    for lado in LADOS:
        grade = pd.read_csv(DATASET / f"grade_{lado}.csv")
        pontos = {
            "mediana das viagens": pontos_mediana(df, grade, lado, origem),
            "centro da célula": (grade["x_centro"].values, grade["y_centro"].values),
        }
        for nome, (x, y) in pontos.items():
            tabela = tempos_vizinhos(g, grade, x, y, fatores)
            resumir(tabela, f"{lado} m, ponto = {nome}")
            if nome == "mediana das viagens":
                tabela.to_csv(DATASET / f"tempos_vizinhos_{lado}.csv", index=False)
