"""
python -m etl.etl              limpa, recorta e salva o Parquet
python -m etl.etl --explorar   só mostra as distribuições
python -m etl.etl --explorar-manhattan   distribuições do Parquet já salvo
python -m etl.etl --tarifa     Q4: tarifa média (USD/km) nos dias de treino
python -m etl.etl --desenhar   desenha os embarques do Parquet
"""

from pathlib import Path

import numpy as np
import pandas as pd

from etl.zonas import EXCLUIDAS, contorno_latlon, dentro, ler_zonas, poligono_manhattan

DATASET = Path(__file__).resolve().parent.parent / "dataset"
ARQUIVO = DATASET / "yellow_tripdata_2016-03.csv"
SAIDA = DATASET / "viagens_manhattan.parquet"

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


def velocidade_mph(df):
    """Velocidade média de cada viagem em milhas por hora."""
    return df["trip_distance"] / (duracao_min(df) / 60)


PERCENTIS = [0.001, 0.01, 0.5, 0.99, 0.995, 0.999, 0.9999]


def explorar_distribuicoes(df):
    """Percentis, extremos e impossíveis de duração, distância e velocidade."""
    series = {
        "duracao (min)": duracao_min(df),
        "trip_distance (mi)": df["trip_distance"],
        "velocidade (mph)": velocidade_mph(df),
    }
    for nome, serie in series.items():
        print(f"\n== {nome} ==")
        print(serie.quantile(PERCENTIS).to_string())
        print(f"min: {serie.min()}")
        print(f"max: {serie.max()}")
        print(f"<= 0: {(serie <= 0).sum():,} ({100 * (serie <= 0).mean():.2f}%)")


def explorar(df):
    """Item 2: distribuições do CSV antes de qualquer corte (só imprime)."""
    explorar_distribuicoes(df)

    embarque = df["tpep_pickup_datetime"]
    print("\n== embarques ==")
    print(f"min: {embarque.min()}")
    print(f"max: {embarque.max()}")
    fora = (embarque.dt.year != 2016) | (embarque.dt.month != 3)
    print(f"fora de mar/2016: {fora.sum():,}")

    zonas = ler_zonas()
    ilhas = contorno_latlon(zonas[zonas["zone"].isin(EXCLUIDAS)])
    print(f"\n== zona excluída ({', '.join(EXCLUIDAS)}) ==")
    for ponta in ("pickup", "dropoff"):
        n = dentro(ilhas, df[f"{ponta}_longitude"].values, df[f"{ponta}_latitude"].values).sum()
        print(f"{ponta}s dentro: {n:,}")


COORDENADAS = ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"]
INICIO_MES = pd.Timestamp("2016-03-01")
FIM_MES = pd.Timestamp("2016-04-01")  # exclusivo


def coordenadas_validas(df):
    c = df[COORDENADAS]
    return c.notna().all(axis=1) & c.ne(0).all(axis=1)


def embarque_no_mes(df):
    # Só o embarque: é ele que cria o pedido. Desembarques em abril são,
    # em geral, corridas que cruzam a meia-noite de 31/03.
    embarque = df["tpep_pickup_datetime"]
    return (embarque >= INICIO_MES) & (embarque < FIM_MES)


def distancia_positiva(df):
    return df["trip_distance"] > 0


def duracao_positiva(df):
    return duracao_min(df) > 0


def embarque_em_manhattan(df):
    return dentro(poligono_manhattan(), df["pickup_longitude"].values, df["pickup_latitude"].values)


def desembarque_em_manhattan(df):
    return dentro(poligono_manhattan(), df["dropoff_longitude"].values, df["dropoff_latitude"].values)


# Limites escolhidos olhando as viagens de Manhattan (--explorar-manhattan, regras 1-6):
# - duração: p99,5% = 40 min e p99,9% = 1.424 min. A contagem cai até ~90 min, fica
#   num patamar baixo e tem um pico de 13.869 viagens entre 23h e 24h (taxímetro
#   esquecido). 2 h é o início desse patamar.
# - distância: Manhattan tem ~13 mi de ponta a ponta. Até 25 mi a velocidade mediana
#   fica em ~24 mph (desvios pelas vias expressas); acima disso sobram ~215 viagens.
# - velocidade: depois dos dois cortes, p99,9% = 36,9 mph e p99,95% = 137,9 mph.
#   Acima de 50 mph a média da viagem é impossível na ilha; essas viagens duram
#   em mediana 17 s (erro de registro do taxímetro).
DURACAO_MAX_MIN = 120
DISTANCIA_MAX_MI = 25
VELOCIDADE_MAX_MPH = 50


def duracao_no_limite(df):
    return duracao_min(df) <= DURACAO_MAX_MIN


def distancia_no_limite(df):
    return df["trip_distance"] <= DISTANCIA_MAX_MI


def velocidade_no_limite(df):
    return velocidade_mph(df) <= VELOCIDADE_MAX_MPH


# Etapas em ordem: (nome, função que devolve a máscara das linhas que FICAM).
# Cada etapa conta só o que ela remove do que sobrou das anteriores.
# Por isso a etapa 6 conta exatamente as viagens que saem de Manhattan.
ETAPAS = [
    ("coordenadas zeradas", coordenadas_validas),
    ("embarque fora de mar/2016", embarque_no_mes),
    ("trip_distance <= 0", distancia_positiva),
    ("duração <= 0", duracao_positiva),
    ("embarque fora de Manhattan", embarque_em_manhattan),
    ("desembarque fora de Manhattan", desembarque_em_manhattan),
    (f"duração > {DURACAO_MAX_MIN} min", duracao_no_limite),
    (f"distância > {DISTANCIA_MAX_MI} mi", distancia_no_limite),
    (f"velocidade > {VELOCIDADE_MAX_MPH} mph", velocidade_no_limite),
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
    # A conta tem que fechar: removidas em cada etapa + restantes = lidas.
    assert relatorio["removidas"].sum() + len(viagens) == lidas
    relatorio.loc["total"] = [lidas - len(viagens), len(viagens)]
    relatorio.insert(1, "% das lidas", 100 * relatorio["removidas"] / lidas)
    return viagens.reset_index(drop=True), relatorio


def imprimir_relatorio(relatorio):
    print(relatorio.to_string(formatters={
        "removidas": "{:,}".format,
        "% das lidas": "{:.3f}%".format,
        "restantes": "{:,}".format,
    }))


DIAS_TREINO = [1, 2, 3, 7, 8, 9, 10]
KM_POR_MILHA = 1.609


def conferir_tarifa(df):
    """Q4: tarifa média em USD/km nos dias de treino (soma das tarifas ÷ soma dos km)."""
    treino = df[df["tpep_pickup_datetime"].dt.day.isin(DIAS_TREINO)]
    pagas = treino[treino["fare_amount"] > 0]
    usd_km = pagas["fare_amount"].sum() / (pagas["trip_distance"] * KM_POR_MILHA).sum()
    print(f"dias de treino {DIAS_TREINO}: {len(treino):,} viagens")
    print(f"fare_amount <= 0: {len(treino) - len(pagas):,} removidas")
    print(f"tarifa: {usd_km:.4f} USD/km")
    return usd_km


def desenhar(df, arquivo=None):
    """Item 6: embarques de uma amostra, com a proporção corrigida para a
    latitude de NY (1 grau de longitude ~ 3/4 de 1 grau de latitude)."""
    import matplotlib.pyplot as plt

    amostra = df.sample(min(50_000, len(df)), random_state=0)
    plt.figure(figsize=(6, 9))
    plt.scatter(amostra["pickup_longitude"], amostra["pickup_latitude"], s=0.5)
    plt.gca().set_aspect(1 / np.cos(np.radians(40.75)))
    plt.title("Embarques (amostra de 50 mil)")
    if arquivo:
        plt.savefig(arquivo, dpi=150, bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Limpeza e recorte de Manhattan.")
    parser.add_argument("--explorar", action="store_true",
                        help="só mostra as distribuições do CSV, sem limpar")
    parser.add_argument("--explorar-manhattan", action="store_true",
                        help="mostra as distribuições do Parquet já salvo (viagens de Manhattan)")
    parser.add_argument("--desenhar", nargs="?", const="", metavar="PNG",
                        help="desenha os embarques do Parquet já salvo (na tela ou num PNG)")
    parser.add_argument("--tarifa", action="store_true",
                        help="Q4: tarifa média em USD/km nos dias de treino (lê o Parquet)")
    args = parser.parse_args()

    if args.tarifa:
        conferir_tarifa(pd.read_parquet(SAIDA))
        raise SystemExit

    if args.explorar_manhattan:
        explorar_distribuicoes(pd.read_parquet(SAIDA))
        raise SystemExit

    if args.desenhar is not None:
        desenhar(pd.read_parquet(SAIDA), args.desenhar or None)
        raise SystemExit

    df = ler()
    print(df.dtypes.to_string(), "\n")
    if args.explorar:
        explorar(df)
        raise SystemExit

    viagens, relatorio = limpar(df)
    imprimir_relatorio(relatorio)
    viagens.to_parquet(SAIDA, index=False)
    print(f"\nSalvo em {SAIDA}")
