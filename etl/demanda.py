"""
Pedidos da grade a partir das viagens (tarefa 1.4, lado dos dados).

python -m etl.demanda    1.4a: converte as viagens em pedidos das grades de 1200 m e 600 m
                         e salva dataset/pedidos_{lado}.parquet
                         1.4b: previsão λ (pedidos médios por célula e passo nos dias de
                         treino, úteis e sábados) em dataset/lambda_{lado}.csv

O simulador só lê a tabela pronta: a conversão de coordenadas em células, que é
trabalho espacial, fica aqui. As células são a chave estável (o id da zona sai na
hora de carregar, com GradeCelulas.zona), como nos pontos e nos tempos.

Tipos compactos: o Parquet guarda dia em int8 e passo e células em int16, que bastam
para guardar, mas não para fazer contas (int16 * 600 "dá a volta" sem aviso). Quem
carrega converte para int64 (no simulador, tarefa 1.4c).
"""
import numpy as np
import pandas as pd

from environment.tempo import PASSOS_POR_DIA, SEGUNDOS_POR_PASSO
from etl.etl import DATASET, DIAS_TREINO, KM_POR_MILHA, SABADOS_TREINO, SAIDA, duracao_min
from etl.grade import LADOS, celula, pontas_em_metros
from etl.zonas import origem_grade


def pedidos_da_grade(df, grade, lado, origem):
    """Uma linha por viagem com as duas pontas em células da grade.
    Devolve (pedidos, removidas): as removidas têm alguma ponta numa célula
    que não está na grade (as células com menos de 20 pontas de treino)."""
    (xe, ye), (xd, yd) = pontas_em_metros(df)
    linha_o, coluna_o = celula(xe, ye, lado, origem)
    linha_d, coluna_d = celula(xd, yd, lado, origem)

    embarque = df["tpep_pickup_datetime"]
    # segundos inteiros e divisão inteira: sem erro de arredondamento na fronteira dos passos
    segundo = (embarque.dt.hour * 3600 + embarque.dt.minute * 60 + embarque.dt.second).values
    todos = pd.DataFrame({
        "dia": embarque.dt.day.values.astype(np.int8),
        "segundo_do_dia": segundo.astype(np.int32),
        "passo": (segundo // SEGUNDOS_POR_PASSO).astype(np.int16),
        "linha_o": linha_o.astype(np.int16), "coluna_o": coluna_o.astype(np.int16),
        "linha_d": linha_d.astype(np.int16), "coluna_d": coluna_d.astype(np.int16),
        "duracao_min": duracao_min(df).values,
        "km": df["trip_distance"].values * KM_POR_MILHA,
    })

    existe = pd.MultiIndex.from_frame(grade[["linha", "coluna"]])
    fica = (pd.MultiIndex.from_arrays([linha_o, coluna_o]).isin(existe)
            & pd.MultiIndex.from_arrays([linha_d, coluna_d]).isin(existe))
    return todos[fica].reset_index(drop=True), todos[~fica].reset_index(drop=True)


# λ (proposta §3.2): pedidos novos esperados com origem na zona i no passo t, pela média
# no mesmo horário nos dias de treino. No simulador:
#   λ = pedidos_medios × 0,15 (fração amostrada) × fator de demanda (1; 0,5 ou 1,5 na RQ2)
TIPOS_DE_DIA = {"util": DIAS_TREINO, "sabado": SABADOS_TREINO}


def previsao_lambda(pedidos, grade, tipos=TIPOS_DE_DIA):
    """Tabela DENSA: toda célula da grade x todo passo (0 a 143) x cada tipo de dia
    (coluna tipo_dia, para não confundir com o dia do mês da tabela de pedidos), com
    pedidos_medios = pedidos com origem na célula naquele passo, somados nos dias do
    tipo, ÷ número de dias do tipo. Zero onde não houve pedido.

    Soma ÷ número de dias, e não groupby().mean(): um dia sem pedido não gera linha
    na contagem, e a média das linhas que existem esqueceria os zeros. Todas as
    viagens, e não a amostra de 15%: a média é a mesma, sem o ruído do sorteio."""
    celulas = list(zip(grade["linha"], grade["coluna"]))
    partes = []
    for tipo, dias in tipos.items():
        do_tipo = pedidos[pedidos["dia"].isin(dias)]
        soma = do_tipo.groupby(["passo", "linha_o", "coluna_o"]).size()
        indice = pd.MultiIndex.from_tuples(
            [(p, l, c) for p in range(PASSOS_POR_DIA) for l, c in celulas],
            names=["passo", "linha_o", "coluna_o"])
        media = soma.reindex(indice, fill_value=0) / len(dias)
        partes.append(media.rename("pedidos_medios").reset_index().assign(tipo_dia=tipo))
    tabela = pd.concat(partes, ignore_index=True)
    tabela = tabela.rename(columns={"linha_o": "linha", "coluna_o": "coluna"})
    return tabela[["tipo_dia", "passo", "linha", "coluna", "pedidos_medios"]]


if __name__ == "__main__":
    df = pd.read_parquet(SAIDA)
    origem = origem_grade()
    print(f"{len(df):,} viagens")

    for lado in LADOS:
        grade = pd.read_csv(DATASET / f"grade_{lado}.csv")
        pedidos, removidas = pedidos_da_grade(df, grade, lado, origem)
        assert len(pedidos) + len(removidas) == len(df)
        assert pedidos["passo"].between(0, PASSOS_POR_DIA - 1).all()

        arquivo = DATASET / f"pedidos_{lado}.parquet"
        pedidos.to_parquet(arquivo, index=False)
        print(f"\n== lado {lado} m ==")
        print(f"pedidos: {len(pedidos):,} | removidas (alguma ponta fora da grade): "
              f"{len(removidas):,} ({len(removidas) / len(df):.3%})")
        print(f"salvo: {arquivo.name} ({arquivo.stat().st_size / 1e6:.0f} MB)")

        lam = previsao_lambda(pedidos, grade)
        assert len(lam) == len(grade) * PASSOS_POR_DIA * len(TIPOS_DE_DIA)
        for tipo, dias in TIPOS_DE_DIA.items():
            # a conta fecha exatamente (todas as viagens, sem amostra):
            # soma de pedidos_medios = pedidos da grade nos dias do tipo ÷ número de dias
            soma = lam.loc[lam["tipo_dia"] == tipo, "pedidos_medios"].sum()
            assert np.isclose(soma, pedidos["dia"].isin(dias).sum() / len(dias), rtol=1e-12, atol=0)
            total = soma * len(dias)
            por_passo = lam[lam["tipo_dia"] == tipo].groupby("passo")["pedidos_medios"].sum()
            zeros = (lam.loc[lam["tipo_dia"] == tipo, "pedidos_medios"] == 0).mean()
            print(f"λ {tipo:>6} ({len(dias)} dias): {total / len(dias):,.0f} pedidos/dia | "
                  f"por passo (grade toda): mín {por_passo.min():,.0f} às {por_passo.idxmin() * 10 // 60:02d}:"
                  f"{por_passo.idxmin() * 10 % 60:02d}, máx {por_passo.max():,.0f} às "
                  f"{por_passo.idxmax() * 10 // 60:02d}:{por_passo.idxmax() * 10 % 60:02d} | "
                  f"(célula, passo) com zero: {zeros:.1%}")
        arquivo = DATASET / f"lambda_{lado}.csv"
        lam.to_csv(arquivo, index=False)
        print(f"salvo: {arquivo.name} ({len(lam):,} linhas)")
