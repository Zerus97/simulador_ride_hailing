from pathlib import Path

import pytest

from environment.env import GradeCelulas, GradeQuadrada, grade_do_csv

DATASET = Path(__file__).resolve().parent.parent / "dataset"


# Escada de cinco células (linha, coluna); ids na ordem (linha, coluna):
#
#   linha 2:   .  .  4
#   linha 1:   1  2  3
#   linha 0:   0  .  .
#
#   0 = (0,0)  1 = (1,0)  2 = (1,1)  3 = (1,2)  4 = (2,2)
ESCADA = [(0, 0), (1, 0), (1, 1), (1, 2), (2, 2)]


@pytest.fixture
def escada():
    return GradeCelulas(ESCADA)


@pytest.mark.parametrize("zona, esperadas", [
    (0, {1, 2}),
    (1, {0, 2}),
    (2, {0, 1, 3, 4}),
    (3, {2, 4}),
    (4, {2, 3}),
])
def test_escada_vizinhas(escada, zona, esperadas):
    assert set(escada.vizinhas(zona)) == esperadas


def test_ids_nao_dependem_da_ordem_de_entrada(escada):
    embaralhada = GradeCelulas([(2, 2), (1, 0), (1, 2), (0, 0), (1, 1)])
    assert [embaralhada.celula(z) for z in embaralhada.zonas()] == \
           [escada.celula(z) for z in escada.zonas()]


def test_exclusao_remove_o_par_nos_dois_sentidos():
    # de propósito na ordem invertida: (2,2) é a zona 4 e (1,1) é a zona 2
    g = GradeCelulas(ESCADA, nao_vizinhos=[((2, 2), (1, 1))])
    assert set(g.vizinhas(2)) == {0, 1, 3}
    assert set(g.vizinhas(4)) == {3}


def test_celula_e_zona_sao_inversas(escada):
    for z in escada.zonas():
        assert escada.zona(escada.celula(z)) == z


def test_ordem_das_vizinhas_5x5():
    # listas, não conjuntos: a ordem das vizinhas importa (o Hotspot desempata por ela).
    # Ordem de DIRECOES: cima, baixo, esquerda, direita e depois as diagonais.
    g = GradeQuadrada(5)
    assert g.vizinhas(0) == [5, 1, 6]                       # baixo, direita, diagonal
    assert g.vizinhas(12) == [7, 17, 11, 13, 6, 8, 16, 18]


def test_excluir_celula_inexistente_da_erro():
    with pytest.raises(ValueError):
        GradeCelulas(ESCADA, nao_vizinhos=[((1, 1), (9, 9))])


def test_par_nao_adjacente_da_erro():
    # (0, 0) e (2, 2) existem na escada, mas não são vizinhas
    with pytest.raises(ValueError):
        GradeCelulas(ESCADA, nao_vizinhos=[((0, 0), (2, 2))])


def test_celula_repetida_da_erro():
    with pytest.raises(ValueError):
        GradeCelulas([(0, 0), (0, 0)])


# --- Leitura do CSV ---

def test_grade_do_csv(tmp_path):
    # linhas em ordem invertida e uma coluna a mais: nenhuma das duas pode atrapalhar
    caminho = tmp_path / "grade.csv"
    caminho.write_text("linha,coluna,embarques\n2,2,9\n1,2,9\n1,1,9\n1,0,9\n0,0,9\n")
    g = grade_do_csv(caminho)
    assert [g.celula(z) for z in g.zonas()] == sorted(ESCADA)


# Sem a Roosevelt Island (pendência 8) e com as células definidas pelos dias de
# treino, com pelo menos MINIMO_PONTAS_TREINO pontas (pendência 9).
@pytest.mark.parametrize("lado, n_zonas", [(1200, 58), (600, 178)])
def test_manhattan_e_conexa(lado, n_zonas):
    caminho = DATASET / f"grade_{lado}.csv"
    if not caminho.exists():
        pytest.skip("dataset fora do repositório")
    g = grade_do_csv(caminho)
    assert len(g.zonas()) == n_zonas
    assert g.conexo()
