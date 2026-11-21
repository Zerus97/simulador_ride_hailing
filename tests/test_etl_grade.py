import pandas as pd

from etl.grade import MINIMO_PONTAS_TREINO, filtrar_por_treino


def test_filtrar_por_treino_na_fronteira():
    assert MINIMO_PONTAS_TREINO == 20  # os casos abaixo foram montados para esse mínimo

    # A = (0, 0): ausente da contagem | B = (0, 1): 19 pontas | C = (0, 2): 20 pontas
    grade = pd.DataFrame({"linha": [0, 0, 0], "coluna": [0, 1, 2], "embarques": [5, 50, 60]})
    pontas = pd.Series([19, 20], index=pd.MultiIndex.from_tuples(
        [(0, 1), (0, 2)], names=["linha", "coluna"]), name="pontas_treino")

    mantidas, removidas = filtrar_por_treino(grade, pontas)

    # C fica: a regra é "pelo menos 20" (pega a troca de >= por >)
    assert list(zip(mantidas["linha"], mantidas["coluna"])) == [(0, 2)]
    assert mantidas["pontas_treino"].tolist() == [20]
    # A e B saem; A com 0, e não NaN (pega o fillna esquecido)
    assert list(zip(removidas["linha"], removidas["coluna"])) == [(0, 0), (0, 1)]
    assert removidas["pontas_treino"].tolist() == [0, 19]
    assert removidas["pontas_treino"].dtype.kind == "i"
