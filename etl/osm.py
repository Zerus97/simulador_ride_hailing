"""
Rede viária de Manhattan (OpenStreetMap) e conferência das velocidades.

python -m etl.osm    baixa a rede (só na primeira vez) e compara, por hora do dia,
                     a duração registrada das viagens de treino com o tempo do
                     caminho mínimo no OSM, com as velocidades padrão do OSMnx.
"""
import numpy as np
import osmnx as ox
import pandas as pd
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from etl.etl import DATASET, DIAS_TREINO, SAIDA, duracao_min
from etl.grade import celula, para_metros, pontas_em_metros
from etl.zonas import poligono_manhattan

REDE = DATASET / "manhattan_drive.graphml"
MARGEM_GRAUS = 0.001  # ~100 m além do contorno, para não cortar as vias da margem (FDR, West Side)
AMOSTRA = 200_000
SEMENTE = 0


def rede():
    """Rede de ruas para carro, com velocidades (km/h) e tempos (s) padrão do OSMnx.

    Fica só o maior componente fortemente conexo: de qualquer nó se chega a
    qualquer outro. Sem isso, um ponto ligado a um beco de mão única não tem
    caminho para as vizinhas (tempo infinito)."""
    if REDE.exists():
        g = ox.load_graphml(REDE)
    else:
        g = ox.graph_from_polygon(poligono_manhattan().buffer(MARGEM_GRAUS), network_type="drive")
        g = ox.add_edge_speeds(g)
        g = ox.add_edge_travel_times(g)
        ox.save_graphml(g, REDE)
    return ox.truncate.largest_component(g, strongly=True)


def matriz_tempos(g):
    """Grafo -> (matriz esparsa de tempos em s, ids dos nós). Entre arestas
    paralelas fica a mais rápida."""
    dg = ox.convert.to_digraph(g, weight="travel_time")
    nos = list(dg.nodes)
    import networkx as nx
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
    """(x, y) em metros de cada célula: mediana dos embarques e desembarques nela."""
    (xe, ye), (xd, yd) = pontas_em_metros(df)
    x, y = np.r_[xe, xd], np.r_[ye, yd]
    linha, coluna = celula(x, y, lado, origem)
    med = pd.DataFrame({"linha": linha, "coluna": coluna, "x": x, "y": y}) \
        .groupby(["linha", "coluna"])[["x", "y"]].median()
    med = med.loc[list(zip(grade["linha"], grade["coluna"]))]
    return med["x"].values, med["y"].values


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
    from etl.grade import LADOS
    from etl.zonas import origem_grade

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
