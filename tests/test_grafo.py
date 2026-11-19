from collections import Counter

import pytest

from environment.env import GradeQuadrada


# Grade 5x5, ids por linha:
#
#  0  1  2  3  4
#  5  6  7  8  9
# 10 11 12 13 14
# 15 16 17 18 19
# 20 21 22 23 24


# --- Testes de exemplo: casos calculados à mão olhando a grade ---
# Comparamos como conjunto para não depender da ordem das vizinhas.

@pytest.mark.parametrize("zona, esperadas", [
    (0,  {1, 5, 6}),                          # canto
    (12, {6, 7, 8, 11, 13, 16, 17, 18}),      # centro
    (24, {18, 19, 23}),                       # canto
    (2,  {1, 3, 6, 7, 8}),                    # borda
])
def test_vizinhas_exemplos(zona, esperadas):
    g = GradeQuadrada(5)
    assert set(g.vizinhas(zona)) == esperadas


# --- Testes de propriedade: regras que valem para todas as zonas ---

@pytest.fixture
def grafo():
    return GradeQuadrada(5)


def test_vizinhanca_simetrica(grafo):
    for a in grafo.zonas():
        for b in grafo.vizinhas(a):
            assert a in grafo.vizinhas(b), f"{b} é vizinha de {a}, mas {a} não é vizinha de {b}"


def test_zona_nao_e_vizinha_de_si_mesma(grafo):
    for z in grafo.zonas():
        assert z not in grafo.vizinhas(z), f"{z} aparece nas próprias vizinhas"


def test_vizinhas_sao_zonas_do_grafo(grafo):
    # Em grafos com zonas removidas (ex.: Manhattan), uma zona pode
    # continuar apontando para uma vizinha que não existe mais.
    zonas = set(grafo.zonas())
    for z in zonas:
        fora = set(grafo.vizinhas(z)) - zonas
        assert not fora, f"{z} tem vizinhas que não são zonas do grafo: {fora}"


def test_contagem_de_vizinhas_5x5(grafo):
    # 4 cantos com 3 vizinhas, 12 bordas com 5 e 9 zonas internas com 8
    contagem = Counter(len(grafo.vizinhas(z)) for z in grafo.zonas())
    assert contagem == {3: 4, 5: 12, 8: 9}
