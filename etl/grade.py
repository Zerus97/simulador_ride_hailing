"""
Células da grade a partir das viagens (tarefa 1.2, lado dos dados).

python -m etl.grade    monta as grades de 1200 m e 600 m, salva os CSVs e as figuras

A grade é uma só para todos os cenários: as células saem dos dias de treino
(DIAS_TREINO), não do dia escolhido para a simulação. Uma célula existe se tem
pelo menos MINIMO_PONTAS_TREINO embarques + desembarques nos dias de treino.
As colunas embarques e desembarques do CSV são do mês inteiro, só como informação.
"""
import numpy as np
import pandas as pd
from pyproj import Transformer

from etl.etl import DATASET, DIAS_TREINO, SAIDA
from etl.zonas import CRS_GRADE, origem_grade

LADOS = [1200, 600]  # lado da célula, em metros
POUCAS_VIAGENS = 10  # limite para listar as células quase vazias

# Mínimo de pontas (embarques + desembarques) nos dias de treino para a célula existir.
# Na grade de 1,2 km, as contagens saltam de 16 para 81; com menos de 20 pontos,
# a mediana que define o ponto da zona não é confiável.
MINIMO_PONTAS_TREINO = 20

# always_xy=True: longitude (x) antes de latitude (y), como no contains_xy.
para_metros = Transformer.from_crs(4326, CRS_GRADE, always_xy=True)
para_latlon = Transformer.from_crs(CRS_GRADE, 4326, always_xy=True)


def pontas_em_metros(df):
    """(x, y) dos embarques e dos desembarques, em metros (UTM 18N)."""
    embarques = para_metros.transform(df["pickup_longitude"].values, df["pickup_latitude"].values)
    desembarques = para_metros.transform(df["dropoff_longitude"].values, df["dropoff_latitude"].values)
    return embarques, desembarques


def celula(x, y, lado, origem):
    """(linha, coluna) de cada ponto: piso da distância à origem dividida pelo lado."""
    x0, y0 = origem
    coluna = np.floor((x - x0) / lado).astype(int)
    linha = np.floor((y - y0) / lado).astype(int)
    return linha, coluna


def contar(x, y, lado, origem, nome):
    linha, coluna = celula(x, y, lado, origem)
    return pd.DataFrame({"linha": linha, "coluna": coluna}).value_counts().rename(nome)


def montar_grade(embarques, desembarques, lado, origem):
    """Uma linha por célula ocupada (pelo menos um embarque ou desembarque)."""
    grade = pd.concat([contar(*embarques, lado, origem, "embarques"),
                       contar(*desembarques, lado, origem, "desembarques")], axis=1)
    grade = grade.fillna(0).astype(int).sort_index().reset_index()

    x0, y0 = origem
    grade["x_centro"] = x0 + (grade["coluna"] + 0.5) * lado
    grade["y_centro"] = y0 + (grade["linha"] + 0.5) * lado
    grade["lon_centro"], grade["lat_centro"] = para_latlon.transform(
        grade["x_centro"].values, grade["y_centro"].values)
    return grade


def pontas_treino(df, lado, origem):
    """Embarques + desembarques por célula, só nos dias de treino."""
    treino = df[df["tpep_pickup_datetime"].dt.day.isin(DIAS_TREINO)]
    embarques, desembarques = pontas_em_metros(treino)
    # fill_value=0: uma célula com só embarques (ou só desembarques) não vira NaN
    return contar(*embarques, lado, origem, "embarques") \
        .add(contar(*desembarques, lado, origem, "desembarques"), fill_value=0) \
        .rename("pontas_treino")


def filtrar_por_treino(grade, pontas):
    """Acrescenta a coluna pontas_treino (0 nas células sem nenhuma) e separa as
    células com menos de MINIMO_PONTAS_TREINO. Devolve (mantidas, removidas)."""
    grade = grade.join(pontas, on=["linha", "coluna"])
    grade["pontas_treino"] = grade["pontas_treino"].fillna(0).astype(int)
    fica = grade["pontas_treino"] >= MINIMO_PONTAS_TREINO
    return grade[fica].reset_index(drop=True), grade[~fica].reset_index(drop=True)


def matriz_embarques(grade):
    """Matriz linha x coluna com os embarques; np.nan nas células sem viagens."""
    matriz = np.full((grade["linha"].max() + 1, grade["coluna"].max() + 1), np.nan)
    matriz[grade["linha"], grade["coluna"]] = grade["embarques"]
    return matriz


def desenhar(grade, lado, arquivo=None):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(5, 9))
    plt.imshow(matriz_embarques(grade), origin="lower")
    plt.colorbar(label="embarques em março")
    plt.title(f"Células de {lado} m ({len(grade)} ocupadas)")
    plt.xlabel("coluna")
    plt.ylabel("linha")
    if arquivo:
        plt.savefig(arquivo, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    df = pd.read_parquet(SAIDA)
    embarques, desembarques = pontas_em_metros(df)
    origem = origem_grade()
    print(f"{len(df):,} viagens | origem: x = {origem[0]:,.1f}  y = {origem[1]:,.1f}")

    for lado in LADOS:
        grade = montar_grade(embarques, desembarques, lado, origem)
        assert (grade[["linha", "coluna"]] >= 0).all().all(), "ponto antes da origem"
        # antes do filtro: toda viagem do mês cai em alguma célula
        assert grade["embarques"].sum() == len(df) == grade["desembarques"].sum()

        grade, removidas = filtrar_por_treino(grade, pontas_treino(df, lado, origem))
        print(f"\n== lado {lado} m: {len(removidas)} células removidas "
              f"(menos de {MINIMO_PONTAS_TREINO} pontas nos dias de treino) ==")
        if len(removidas):
            removidas["pontas_mes"] = removidas["embarques"] + removidas["desembarques"]
            print(removidas[["linha", "coluna", "pontas_treino", "pontas_mes", "embarques",
                             "desembarques", "lat_centro", "lon_centro"]]
                  .round({"lat_centro": 4, "lon_centro": 4}).to_string(index=False))

        csv = DATASET / f"grade_{lado}.csv"
        png = DATASET / f"grade_{lado}.png"
        grade.to_csv(csv, index=False)
        desenhar(grade, lado, png)

        pontas = grade["embarques"] + grade["desembarques"]
        poucas = grade[pontas < POUCAS_VIAGENS]
        linhas, colunas = grade["linha"].max() + 1, grade["coluna"].max() + 1
        print(f"\n== lado {lado} m ==")
        print(f"células ocupadas: {len(grade)} (de {linhas} x {colunas} = {linhas * colunas} no retângulo)")
        print(f"com menos de {POUCAS_VIAGENS} viagens (embarques + desembarques): {len(poucas)}")
        if len(poucas):
            print(poucas[["linha", "coluna", "embarques", "desembarques",
                          "lat_centro", "lon_centro"]].to_string(index=False))
        print(f"salvo: {csv.name}, {png.name}")
