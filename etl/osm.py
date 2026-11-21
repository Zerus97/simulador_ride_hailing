"""
Rede viária de Manhattan (OpenStreetMap), pontos das zonas e fatores por hora.

python -m etl.osm --rede      1.3a: baixa a rede na data fixa (se ainda não houver),
                              confere e mostra os metadados
python -m etl.osm --pontos    1.3b: ponto de cada célula ligado ao cruzamento mais próximo
python -m etl.osm --fatores   1.3c: fator por hora (dias úteis e sábados) e validação
python -m etl.osm --diagnosticos  1.3c: o fator geral serve para cada grupo de viagens?
python -m etl.osm --tempos    1.3d: minutos e km entre células adjacentes, pares acima
                              de 15 min (não vizinhos) e conexidade do tabuleiro
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

from etl.etl import (DATASET, DIAS_TREINO, DIAS_VALIDACAO, KM_POR_MILHA, SABADOS_TREINO,
                     SAIDA, duracao_min)
from etl.grade import celula, para_latlon, para_metros, pontas_em_metros
from environment.env import grade_do_csv
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


FAIXA_VALIDACAO = (0.9, 1.1)  # registrado ÷ calibrado aceitável em cada hora


def viagens_dias(df, dias):
    """Todas as viagens com embarque nos `dias` de março."""
    return df[df["tpep_pickup_datetime"].dt.day.isin(dias)]


def amostra_dias(df, dias):
    """Viagens com embarque nos `dias` de março; no máximo AMOSTRA, sorteadas com a SEMENTE."""
    viagens = viagens_dias(df, dias)
    return viagens.sample(min(AMOSTRA, len(viagens)), random_state=SEMENTE)


# O fator só é usado nos movimentos entre vizinhas, então é calibrado com viagens
# desse tamanho: distância registrada entre 1 e 2,5 vezes o lado da célula.
FAIXA_VIZINHAS = (1.0, 2.5)


def na_faixa_vizinhas(tempos, lado):
    """Só as viagens com distância registrada na FAIXA_VIZINHAS da grade de `lado` metros."""
    minimo, maximo = (m * lado / 1000 for m in FAIXA_VIZINHAS)
    return tempos[tempos["dist_km"].between(minimo, maximo)]


def tempos_viagens(viagens, g, matriz, nos):
    """Para cada viagem: hora do embarque, duração registrada (min), tempo do OSM (min),
    distância registrada (km), latitude do embarque (para a região) e origem/destino
    (posições dos cruzamentos em `nos`, para recalcular o tempo em outra rede).
    Só as válidas: nós diferentes e caminho finito."""
    origem, _ = no_mais_proximo(g, nos, viagens["pickup_longitude"].values, viagens["pickup_latitude"].values)
    destino, _ = no_mais_proximo(g, nos, viagens["dropoff_longitude"].values, viagens["dropoff_latitude"].values)
    tempos = pd.DataFrame({
        "dia": viagens["tpep_pickup_datetime"].dt.day.values,
        "hora": viagens["tpep_pickup_datetime"].dt.hour.values,
        "registrado_min": duracao_min(viagens).values,
        "osm_min": tempos_osm(matriz, origem, destino) / 60,
        "dist_km": viagens["trip_distance"].values * KM_POR_MILHA,
        "lat_embarque": viagens["pickup_latitude"].values,
        "origem": origem,
        "destino": destino,
    })
    validas = (origem != destino) & np.isfinite(tempos["osm_min"]) & (tempos["osm_min"] > 0)
    return tempos[validas].reset_index(drop=True)


def fatores_por_hora(tempos):
    """Por hora: viagens e fator = soma das durações registradas ÷ soma dos tempos do OSM.

    Soma ÷ soma (e não a mediana das razões), como na tarifa da Q4: assim o tempo
    total simulado é igual ao tempo total registrado em cada hora."""
    por_hora = tempos.groupby("hora").agg(
        viagens=("osm_min", "size"),
        registrado_min=("registrado_min", "sum"),
        osm_min=("osm_min", "sum"),
    )
    por_hora["fator"] = por_hora["registrado_min"] / por_hora["osm_min"]
    return por_hora[["viagens", "fator"]]


def razao_por_grupo(tempos, fatores, grupo):
    """Por grupo: viagens e soma das durações registradas ÷ soma dos tempos calibrados
    (OSM × fator da hora). Perto de 1 = o fator serve para o grupo; marca os grupos
    fora de FAIXA_VALIDACAO.
    `grupo`: nome de uma coluna de `tempos`, ou lista de colunas (ex.: ["regiao", "hora"])."""
    dados = tempos.assign(calibrado_min=tempos["osm_min"] * tempos["hora"].map(fatores["fator"]))
    por_grupo = dados.groupby(grupo, observed=True).agg(
        viagens=("registrado_min", "size"),
        registrado_min=("registrado_min", "sum"),
        calibrado_min=("calibrado_min", "sum"),
    )
    por_grupo["razao"] = por_grupo["registrado_min"] / por_grupo["calibrado_min"]
    baixo, alto = FAIXA_VALIDACAO
    por_grupo["fora"] = (por_grupo["razao"] < baixo) | (por_grupo["razao"] > alto)
    return por_grupo[["viagens", "razao", "fora"]]


def validar(tempos, fatores):
    """Por hora: soma das durações registradas ÷ soma dos tempos calibrados (OSM × fator)."""
    return razao_por_grupo(tempos, fatores, "hora")


FAIXAS_KM = [0, 1, 3, 6, np.inf]
ROTULOS_KM = ["<1", "1-3", "3-6", ">6"]


def com_faixa_km(tempos):
    return tempos.assign(faixa_km=pd.cut(tempos["dist_km"], FAIXAS_KM, labels=ROTULOS_KM))


def diagnostico_viagens_curtas(tempos, fatores):
    """Diagnóstico 1 (pendência 2): o fator geral serve para viagens curtas?
    Os reposicionamentos têm 1,2 a 1,7 km, então a faixa que importa é a 1-3."""
    tempos = com_faixa_km(tempos)
    por_faixa = razao_por_grupo(tempos, fatores, "faixa_km")
    curta_por_hora = razao_por_grupo(tempos[tempos["faixa_km"] == "1-3"], fatores, "hora")
    return por_faixa, curta_por_hora


# Sem a via principal, os ramais não levam a lugar nenhum: saem junto.
EXPRESSAS_E_RAMAIS = VIAS_EXPRESSAS | {t + "_link" for t in VIAS_EXPRESSAS}
LIMIAR_USA_EXPRESSA = 1.10  # sem as expressas, o caminho fica mais de 10% mais lento


def rede_sem_expressas(g):
    g_sem = g.copy()
    g_sem.remove_edges_from([(u, v, k) for u, v, k, d in g.edges(keys=True, data=True)
                             if tipo_via(d["highway"]) in EXPRESSAS_E_RAMAIS])
    return g_sem


def diagnostico_expressas(tempos, fatores, g, nos):
    """Diagnóstico 2 (o teste dos 40 mph): o fator geral serve para as viagens que
    usam as vias expressas? Compara dentro da mesma faixa de distância, porque
    viagens longas já têm razão menor por natureza (menos semáforos por km)."""
    matriz_sem, nos_sem = matriz_tempos(rede_sem_expressas(g))
    assert nos_sem == nos  # mesmos nós, na mesma ordem: origem e destino continuam valendo
    osm_sem = tempos_osm(matriz_sem, tempos["origem"].values, tempos["destino"].values) / 60
    # tempo infinito: cruzamento que só se alcança pela expressa também conta como "usa"
    usa = ~np.isfinite(osm_sem) | (osm_sem > LIMIAR_USA_EXPRESSA * tempos["osm_min"].values)
    tempos = com_faixa_km(tempos).assign(usa_expressa=usa)
    por_grupo = razao_por_grupo(tempos, fatores, ["faixa_km", "usa_expressa"])
    fracao = tempos.groupby("faixa_km", observed=True)["usa_expressa"].mean()
    return por_grupo, fracao, tempos["usa_expressa"].mean()


# Rua 59 aproximada: as ruas são inclinadas, e a 59 vai de 40,763 (Lexington) a
# 40,768 (Columbus Circle). Para um diagnóstico, basta.
LATITUDE_RUA_59 = 40.765


def diagnostico_regiao(tempos, fatores):
    """Diagnóstico 3 (pendência 3): o trânsito abaixo da Rua 59 (Midtown e Downtown)
    pede um fator por região? Por região (do embarque) e por hora."""
    tempos = tempos.assign(regiao=np.where(tempos["lat_embarque"] < LATITUDE_RUA_59, "abaixo_59", "acima_59"))
    return razao_por_grupo(tempos, fatores, "regiao"), razao_por_grupo(tempos, fatores, ["regiao", "hora"])


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


# --- 1.3d: minutos e km entre células adjacentes, e a regra dos 15 min ---

# 15 min já viram 2 passos de 10 min (floor(1,5 + 0,5) = 2), então o limite é >= 15.
LIMITE_VIZINHA_MIN = 15


def tempos_adjacentes(g, lado):
    """Uma linha por par ORDENADO de células adjacentes (a vizinhança da GradeCelulas,
    sem exclusões): tempo pelo OSM (min, sem fator) e km do mesmo caminho mais rápido.

    Os pontos de cada célula vêm do pontos_{lado}.csv (1.3b). Os km são do caminho que
    deu o tempo, porque o carro segue o caminho mais rápido, e não o mais curto."""
    celulas = grade_do_csv(DATASET / f"grade_{lado}.csv")
    no_de = pd.read_csv(DATASET / f"pontos_{lado}.csv").set_index(["linha", "coluna"])["no_osm"]
    pares = [(celulas.celula(a), celulas.celula(b))
             for a in celulas.zonas() for b in celulas.vizinhas(a)]

    matriz, nos = matriz_tempos(g)
    posicao = {n: i for i, n in enumerate(nos)}
    origens = sorted({a for a, _ in pares})
    linha_de = {c: i for i, c in enumerate(origens)}
    dist = dijkstra(matriz, indices=[posicao[no_de[c]] for c in origens])  # s, origem x nó

    linhas = []
    for a, b in pares:
        no_a, no_b = int(no_de[a]), int(no_de[b])
        osm_min = dist[linha_de[a], posicao[no_b]] / 60
        rota = nx.shortest_path(g, no_a, no_b, weight="travel_time") if no_a != no_b else [no_a]
        trechos = ox.routing.route_to_gdf(g, rota, weight="travel_time") if len(rota) > 1 else None
        km = trechos["length"].sum() / 1000 if trechos is not None else 0.0
        rota_min = trechos["travel_time"].sum() / 60 if trechos is not None else 0.0
        # o caminho do networkx é o mesmo do Dijkstra da matriz (ou um empate de tempo)
        assert abs(rota_min - osm_min) < 1e-6, f"{a}->{b}: rota {rota_min} != Dijkstra {osm_min}"
        linhas.append({"linha_a": a[0], "coluna_a": a[1], "linha_b": b[0], "coluna_b": b[1],
                       "osm_min": osm_min, "km": km})
    return pd.DataFrame(linhas)


def conferir_tempos(tempos, lado):
    print(f"\n== {lado} m: {len(tempos)} pares ordenados de células adjacentes ==")
    zero = tempos[tempos["osm_min"] == 0]
    print(f"pares com osm_min = 0 (mesmo cruzamento): {len(zero)}")
    if len(zero):
        print(zero.to_string(index=False))
    minimo, maximo = (m * lado / 1000 for m in FAIXA_VIZINHAS)
    dentro = tempos["km"].between(minimo, maximo).mean()
    print(f"km: mínimo {tempos['km'].min():.2f} | mediana {tempos['km'].median():.2f} | "
          f"máximo {tempos['km'].max():.2f} | dentro da faixa do fator ({minimo:.1f} a {maximo:.1f} km): {dentro:.1%}")
    print(f"osm_min: mediana {tempos['osm_min'].median():.2f} | máximo {tempos['osm_min'].max():.2f}")


def nao_vizinhos(tempos, maior_fator):
    """Pares (não ordenados) de células adjacentes que deixam de ser vizinhas.

    O tempo calibrado é OSM × fator(hora), e o fator é o mesmo para todos os pares numa
    hora; logo a pior hora é sempre a do maior fator, e a regra "abaixo de 15 min em
    todas as horas" vira um teste só. Por causa das mãos únicas, olha os dois sentidos:
    se um passa do limite, o par sai nos dois (simetria)."""
    ida = tempos.rename(columns={"osm_min": "osm_ida_min"})
    volta = tempos.rename(columns={"linha_a": "linha_b", "coluna_a": "coluna_b",
                                   "linha_b": "linha_a", "coluna_b": "coluna_a",
                                   "osm_min": "osm_volta_min"})
    pares = ida.merge(volta[["linha_a", "coluna_a", "linha_b", "coluna_b", "osm_volta_min"]],
                      on=["linha_a", "coluna_a", "linha_b", "coluna_b"])
    # cada par não ordenado uma vez só: a célula a vem antes da b
    pares = pares[[(la, ca) < (lb, cb) for la, ca, lb, cb in
                   zip(pares["linha_a"], pares["coluna_a"], pares["linha_b"], pares["coluna_b"])]]
    pares = pares.assign(calibrado_pior_min=pares[["osm_ida_min", "osm_volta_min"]].max(axis=1) * maior_fator)
    excluidos = pares[pares["calibrado_pior_min"] >= LIMITE_VIZINHA_MIN]
    return excluidos[["linha_a", "coluna_a", "linha_b", "coluna_b",
                      "osm_ida_min", "osm_volta_min", "calibrado_pior_min"]].reset_index(drop=True)


def componentes(grafo):
    """Zonas de cada componente conexo, do maior para o menor (para o diagnóstico)."""
    vistas, saida = set(), []
    for z in grafo.zonas():
        if z in vistas:
            continue
        comp, pilha = [], [z]
        vistas.add(z)
        while pilha:
            atual = pilha.pop()
            comp.append(atual)
            for v in grafo.vizinhas(atual):
                if v not in vistas:
                    vistas.add(v)
                    pilha.append(v)
        saida.append(comp)
    return sorted(saida, key=len, reverse=True)


if __name__ == "__main__":
    import argparse

    from etl.grade import LADOS
    from etl.zonas import origem_grade

    parser = argparse.ArgumentParser(description="Rede viária de Manhattan (OpenStreetMap).")
    parser.add_argument("--rede", action="store_true",
                        help="1.3a: só baixa (se preciso) e confere a rede, e mostra os metadados")
    parser.add_argument("--pontos", action="store_true",
                        help="1.3b: ponto de cada célula ligado ao cruzamento mais próximo")
    parser.add_argument("--fatores", action="store_true",
                        help="1.3c: fator por hora (dias úteis e sábados) e validação fora da amostra")
    parser.add_argument("--tempos", action="store_true",
                        help="1.3d: minutos e km entre células adjacentes, regra dos 15 min e conexidade")
    parser.add_argument("--diagnosticos", action="store_true",
                        help="1.3c: o fator geral serve para cada grupo de viagens? (dias de validação)")
    args = parser.parse_args()

    if args.diagnosticos:
        df = pd.read_parquet(SAIDA)
        g = rede()
        matriz, nos = matriz_tempos(g)
        f = pd.read_csv(DATASET / "fatores_hora.csv")
        # os diagnósticos comparam com o fator geral (todas as distâncias) dos dias úteis
        f = f[(f["dia"] == "util") & (f["lado"] == f["lado"].iloc[0])]
        fatores = f.set_index("hora")[["fator_geral"]].rename(columns={"fator_geral": "fator"})
        tempos = tempos_viagens(amostra_dias(df, DIAS_VALIDACAO), g, matriz, nos)
        print(f"dias de validação {DIAS_VALIDACAO}: {len(tempos):,} viagens válidas na amostra")

        por_faixa, curta_por_hora = diagnostico_viagens_curtas(tempos, fatores)
        print("\n== Diagnóstico 1: viagens curtas (registrado ÷ calibrado) ==")
        print(por_faixa.round(3).to_string())
        print("\nfaixa 1-3 km, por hora:")
        print(curta_por_hora.round(3).to_string())
        fora = list(curta_por_hora.index[curta_por_hora["fora"]])
        print(f"horas da faixa 1-3 km fora de {FAIXA_VALIDACAO}: {fora or 'nenhuma'}")

        por_grupo, fracao, fracao_geral = diagnostico_expressas(tempos, fatores, g, nos)
        print("\n== Diagnóstico 2: vias expressas (registrado ÷ calibrado) ==")
        print(f"usam as expressas: {fracao_geral:.1%} das viagens | por faixa: "
              + ", ".join(f"{faixa} {v:.1%}" for faixa, v in fracao.items()))
        print(por_grupo.round(3).to_string())
        razao = por_grupo["razao"].unstack("usa_expressa")
        for faixa in ["3-6", ">6"]:
            if faixa in razao.index and razao.loc[faixa].notna().all():
                dif = razao.loc[faixa, True] / razao.loc[faixa, False] - 1
                print(f"faixa {faixa}: usa ÷ não usa = {dif:+.1%}")

        por_regiao, por_regiao_hora = diagnostico_regiao(tempos, fatores)
        print("\n== Diagnóstico 3: região, abaixo e acima da Rua 59 (registrado ÷ calibrado) ==")
        print(por_regiao.round(3).to_string())
        tabela = por_regiao_hora.unstack("regiao")
        tabela.columns = [f"{r}_{c}" for c, r in tabela.columns]
        print(tabela[[f"{r}_{c}" for r in ("abaixo_59", "acima_59") for c in ("viagens", "razao", "fora")]]
              .round(3).to_string())
        fora = por_regiao_hora[por_regiao_hora["fora"]]
        print(f"(região, hora) fora de {FAIXA_VALIDACAO}: {list(fora.index) or 'nenhuma'}")
        raise SystemExit

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

    if args.fatores:
        df = pd.read_parquet(SAIDA)
        g = rede()
        matriz, nos = matriz_tempos(g)  # a parte cara: monta uma vez só
        # Todas as viagens, sem amostra: o custo está no Dijkstra (um por cruzamento de
        # origem), que é praticamente o mesmo; e as faixas estreitas precisam de volume.
        conjuntos = {"util": DIAS_TREINO, "sabado": SABADOS_TREINO, "validacao": DIAS_VALIDACAO}
        tempos = {nome: tempos_viagens(viagens_dias(df, dias), g, matriz, nos)
                  for nome, dias in conjuntos.items()}
        for nome, dias in conjuntos.items():
            print(f"{nome:>9} (dias {dias}): {len(tempos[nome]):,} viagens válidas")

        geral = {dia: fatores_por_hora(tempos[dia])["fator"] for dia in ("util", "sabado")}
        linhas = []
        validacao = {}
        a, b = SABADOS_TREINO
        for lado in LADOS:
            faixa = {nome: na_faixa_vizinhas(t, lado) for nome, t in tempos.items()}
            for dia in ("util", "sabado"):
                f = fatores_por_hora(faixa[dia]).reset_index()
                linhas.append(f.assign(lado=lado, dia=dia, fator_geral=f["hora"].map(geral[dia])))
            fat_util = fatores_por_hora(faixa["util"])
            sab = {d: faixa["sabado"][faixa["sabado"]["dia"] == d] for d in SABADOS_TREINO}
            validacao[lado] = {
                "util: 14–16/mar": validar(faixa["validacao"], fat_util),
                f"sábado: fatores {a} -> valida {b}": validar(sab[b], fatores_por_hora(sab[a])),
                f"sábado: fatores {b} -> valida {a}": validar(sab[a], fatores_por_hora(sab[b])),
            }

        saida = pd.concat(linhas)[["lado", "dia", "hora", "viagens", "fator", "fator_geral"]]
        saida.to_csv(DATASET / "fatores_hora.csv", index=False)

        print(f"\n== Fatores na faixa das vizinhas ({FAIXA_VIZINHAS[0]} a {FAIXA_VIZINHAS[1]} x o lado) ==")
        for (lado, dia), f in saida.groupby(["lado", "dia"], sort=False):
            menor = f.loc[f["viagens"].idxmin()]
            print(f"{lado} m, {dia:>6}: {f['viagens'].sum():>9,} viagens | hora com menos: "
                  f"{int(menor['hora']):>2}h com {int(menor['viagens']):,} | fator de {f['fator'].min():.2f} a {f['fator'].max():.2f}")

        print(f"\n== Validação na faixa (registrado ÷ calibrado; fora de {FAIXA_VALIDACAO}) ==")
        for lado, vals in validacao.items():
            for nome, v in vals.items():
                fora = v[v["fora"]]
                desc = ", ".join(f"{h}h ({r:.3f}, {n:,} viagens)" for h, r, n in
                                 zip(fora.index, fora["razao"], fora["viagens"])) or "nenhuma"
                print(f"{lado} m, {nome:<28} de {v['razao'].min():.3f} a {v['razao'].max():.3f} | fora: {desc}")

        u = saida[(saida["lado"] == 1200) & (saida["dia"] == "util")].set_index("hora")
        print("\n== 1200 m, dias úteis: fator da faixa ÷ fator geral ==")
        print((u["fator"] / u["fator_geral"]).round(3).to_frame("faixa ÷ geral").T.to_string())
        print(f"\nsalvo: fatores_hora.csv ({len(saida)} linhas)")
        raise SystemExit

    if args.tempos:
        g = rede()
        fatores = pd.read_csv(DATASET / "fatores_hora.csv")
        for lado in LADOS:
            tempos = tempos_adjacentes(g, lado)
            tempos.to_csv(DATASET / f"tempos_{lado}.csv", index=False)
            conferir_tempos(tempos, lado)

            maior = fatores.loc[fatores["lado"] == lado, "fator"].max()
            excluidos = nao_vizinhos(tempos, maior)
            excluidos.to_csv(DATASET / f"nao_vizinhos_{lado}.csv", index=False)
            print(f"maior fator: {maior:.2f} -> exclui se o tempo pelo OSM >= "
                  f"{LIMITE_VIZINHA_MIN} / {maior:.2f} = {LIMITE_VIZINHA_MIN / maior:.2f} min, em algum sentido")
            print(f"pares excluídos: {len(excluidos)} de {len(tempos) // 2}")
            if len(excluidos):
                print(excluidos.round(2).to_string(index=False))

            tabuleiro = grade_do_csv(
                DATASET / f"grade_{lado}.csv",
                nao_vizinhos=[((r.linha_a, r.coluna_a), (r.linha_b, r.coluna_b)) for r in excluidos.itertuples()])
            sem_vizinha = [tabuleiro.celula(z) for z in tabuleiro.zonas() if not tabuleiro.vizinhas(z)]
            print(f"tabuleiro: {len(tabuleiro.zonas())} zonas | conexo = {tabuleiro.conexo()} | "
                  f"zonas sem vizinha: {sem_vizinha or 'nenhuma'}")
            if not tabuleiro.conexo():
                for comp in componentes(tabuleiro)[1:]:
                    print(f"  isoladas do resto: {[tabuleiro.celula(z) for z in comp]}")
            print(f"salvo: tempos_{lado}.csv, nao_vizinhos_{lado}.csv")
        raise SystemExit

    parser.print_help()
