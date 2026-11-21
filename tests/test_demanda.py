import pandas as pd
import pytest

from etl.demanda import pedidos_da_grade, previsao_lambda
from etl.grade import para_latlon

# Células de 1000 m com origem em (583000, 4506000), em metros (UTM 18N).
# A grade tem (0, 0) e (0, 1); a (5, 5) fica fora dela.
ORIGEM = (583_000, 4_506_000)
LADO = 1000
GRADE = pd.DataFrame({"linha": [0, 0], "coluna": [0, 1]})
EM_00 = (583_500, 4_506_500)   # centro da (0, 0)
EM_01 = (584_500, 4_506_500)   # centro da (0, 1)
FORA = (588_500, 4_511_500)    # centro da (5, 5)


def viagens(lista):
    """lista de (embarque 'AAAA-MM-DD hh:mm:ss', ponto do embarque, ponto do desembarque)."""
    (xe, ye), (xd, yd) = zip(*[e for _, e, _ in lista]), zip(*[d for _, _, d in lista])
    lon_e, lat_e = para_latlon.transform(list(xe), list(ye))
    lon_d, lat_d = para_latlon.transform(list(xd), list(yd))
    embarque = pd.to_datetime([h for h, _, _ in lista])
    return pd.DataFrame({
        "tpep_pickup_datetime": embarque,
        "tpep_dropoff_datetime": embarque + pd.Timedelta(minutes=12),
        "pickup_longitude": lon_e, "pickup_latitude": lat_e,
        "dropoff_longitude": lon_d, "dropoff_latitude": lat_d,
        "trip_distance": 1.0,
    })


@pytest.mark.parametrize("horario, passo", [
    ("00:09:59", 0),
    ("00:10:00", 1),
    ("23:59:59", 143),
])
def test_fronteira_dos_passos(horario, passo):
    pedidos, _ = pedidos_da_grade(viagens([(f"2016-03-07 {horario}", EM_00, EM_01)]), GRADE, LADO, ORIGEM)
    assert pedidos["passo"].tolist() == [passo]
    h, m, s = map(int, horario.split(":"))
    assert pedidos["segundo_do_dia"].tolist() == [h * 3600 + m * 60 + s]


def test_viagem_com_ponta_fora_da_grade_sai():
    pedidos, removidas = pedidos_da_grade(viagens([
        ("2016-03-07 08:00:00", EM_00, EM_01),  # as duas pontas dentro: fica
        ("2016-03-07 08:00:00", EM_00, FORA),   # desembarque fora: sai
    ]), GRADE, LADO, ORIGEM)

    assert len(pedidos) == 1 and len(removidas) == 1
    assert pedidos.loc[0, ["linha_o", "coluna_o", "linha_d", "coluna_d"]].tolist() == [0, 0, 0, 1]
    assert removidas.loc[0, ["linha_d", "coluna_d"]].tolist() == [5, 5]
    assert pedidos.loc[0, "dia"] == 7
    assert pedidos.loc[0, "duracao_min"] == pytest.approx(12)
    assert pedidos.loc[0, "km"] == pytest.approx(1.609)


# --- previsão λ (1.4b) ---

def pedidos_de_mentira(linhas):
    """linhas: (dia, passo, linha_o, coluna_o); o resto das colunas não importa para o λ."""
    return pd.DataFrame(linhas, columns=["dia", "passo", "linha_o", "coluna_o"])


def test_lambda_conta_os_dias_sem_pedido():
    # dois dias de treino (1 e 2); 3 pedidos na (0, 0) no passo 5 do dia 1, nenhum no dia 2,
    # e mais um no dia 23, fora dos dias de treino
    pedidos = pedidos_de_mentira([(1, 5, 0, 0)] * 3 + [(23, 5, 0, 0)])
    lam = previsao_lambda(pedidos, GRADE, tipos={"util": [1, 2]})

    def valor(linha, coluna):
        return lam.query(f"passo == 5 and linha == {linha} and coluna == {coluna}")["pedidos_medios"].tolist()

    assert valor(0, 0) == [1.5]  # (3 + 0) ÷ 2; o groupby().mean() daria 3,0; com o dia 23, 2,0
    assert valor(0, 1) == [0.0]  # célula sem nenhum pedido: aparece, com 0


def test_lambda_tabela_densa_e_so_dias_do_tipo():
    pedidos = pedidos_de_mentira([(1, 5, 0, 0), (2, 5, 0, 0), (5, 7, 0, 1), (23, 5, 0, 0)])
    lam = previsao_lambda(pedidos, GRADE, tipos={"util": [1, 2], "sabado": [5]})

    # toda célula × todo passo × cada tipo, com zero onde não houve pedido
    assert len(lam) == 2 * 144 * 2
    assert lam["pedidos_medios"].notna().all()
    assert (lam["pedidos_medios"] == 0).sum() == len(lam) - 2
    util = lam.query("tipo_dia == 'util' and passo == 5 and linha == 0 and coluna == 0")
    sabado = lam.query("tipo_dia == 'sabado' and passo == 7 and linha == 0 and coluna == 1")
    assert util["pedidos_medios"].tolist() == [1.0]    # dias 1 e 2; o dia 23 não entra
    assert sabado["pedidos_medios"].tolist() == [1.0]
