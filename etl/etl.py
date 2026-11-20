"""Preparação das viagens de táxi de Manhattan para o simulador.

Os dados ficam em dataset/ (fora do git, pelo .gitignore).
"""
from pathlib import Path

import pandas as pd

ARQUIVO = Path(__file__).resolve().parent.parent / "dataset" / "yellow_tripdata_2016-03.csv"

# Item 1: das 19 colunas do arquivo, a preparação usa só estas 8.
COLUNAS = [
    "tpep_pickup_datetime",   # embarque: passo de criação do pedido, dia, janela 4h-24h
    "tpep_dropoff_datetime",  # desembarque: duração da viagem
    "pickup_longitude",       # zona de origem
    "pickup_latitude",
    "dropoff_longitude",      # zona de destino
    "dropoff_latitude",
    "trip_distance",          # tarifa (milhas; x 1,609 para km)
    "fare_amount",            # só para a conferência da tarifa (Q4)
]
COLUNAS_DATA = ["tpep_pickup_datetime", "tpep_dropoff_datetime"]
FORMATO_DATA = "%Y-%m-%d %H:%M:%S"


def ler(caminho=ARQUIVO):
    """Lê o CSV com só as colunas necessárias e as datas já convertidas."""
    df = pd.read_csv(caminho, usecols=COLUNAS,
                     parse_dates=COLUNAS_DATA, date_format=FORMATO_DATA)
    return df[COLUNAS]


def duracao_min(df):
    """Duração de cada viagem em minutos."""
    return (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds() / 60


def explorar(df):
    """Item 2: distribuições antes de qualquer corte (só imprime)."""
    percentis = [0.001, 0.01, 0.5, 0.99, 0.999]
    series = {"duracao (min)": duracao_min(df), "trip_distance (mi)": df["trip_distance"]}
    for nome, serie in series.items():
        print(f"\n== {nome} ==")
        print(serie.quantile(percentis).to_string())
        print(f"min: {serie.min()}")
        print(f"max: {serie.max()}")
        print(f"<= 0: {(serie <= 0).sum():,} ({100 * (serie <= 0).mean():.2f}%)")

    embarque = df["tpep_pickup_datetime"]
    print("\n== embarques ==")
    print(f"min: {embarque.min()}")
    print(f"max: {embarque.max()}")
    fora = (embarque.dt.year != 2016) | (embarque.dt.month != 3)
    print(f"fora de mar/2016: {fora.sum():,}")


COORDENADAS = ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"]
INICIO_MES = pd.Timestamp("2016-03-01")
FIM_MES = pd.Timestamp("2016-04-01")  # exclusivo


def coordenadas_validas(df):
    c = df[COORDENADAS]
    return c.notna().all(axis=1) & c.ne(0).all(axis=1)


def datas_no_mes(df):
    return (df[COLUNAS_DATA] >= INICIO_MES).all(axis=1) & (df[COLUNAS_DATA] < FIM_MES).all(axis=1)


def distancia_positiva(df):
    return df["trip_distance"] > 0


def duracao_positiva(df):
    return duracao_min(df) > 0


# Etapas em ordem: (nome, função que devolve a máscara das linhas que FICAM).
# Cada etapa conta só o que ela remove do que sobrou das anteriores.
ETAPAS = [
    ("coordenadas zeradas", coordenadas_validas),
    ("datas fora de mar/2016", datas_no_mes),
    ("trip_distance <= 0", distancia_positiva),
    ("duração <= 0", duracao_positiva),
]


def limpar(df):
    """Devolve (viagens que sobraram, relatório).

    O relatório é um DataFrame com uma linha por etapa: quantas linhas
    ela removeu, quanto isso é das lidas e quantas sobraram depois dela.
    """
    lidas = len(df)
    linhas = [{"etapa": "lidas", "removidas": 0, "restantes": lidas}]
    viagens = df
    for nome, manter in ETAPAS:
        mascara = manter(viagens)
        removidas = int((~mascara).sum())
        viagens = viagens[mascara]
        linhas.append({"etapa": nome, "removidas": removidas, "restantes": len(viagens)})

    relatorio = pd.DataFrame(linhas).set_index("etapa")
    relatorio.loc["total"] = [lidas - len(viagens), len(viagens)]
    relatorio.insert(1, "% das lidas", 100 * relatorio["removidas"] / lidas)
    return viagens.reset_index(drop=True), relatorio


def imprimir_relatorio(relatorio):
    print(relatorio.to_string(formatters={
        "removidas": "{:,}".format,
        "% das lidas": "{:.2f}%".format,
        "restantes": "{:,}".format,
    }))


if __name__ == "__main__":
    import sys

    df = ler()
    print(df.dtypes.to_string(), "\n")
    if "--explorar" in sys.argv:
        explorar(df)
        sys.exit()
    viagens, relatorio = limpar(df)
    imprimir_relatorio(relatorio)
