"""O calendário de março/2016 (environment/calendario.py)."""
import datetime

import pytest

from environment.calendario import (DIAS_AVALIACAO, DIAS_AVALIACAO_EXTRAS, DIAS_EXCLUIDOS, DIAS_TREINO,
                                    DIAS_VALIDACAO, SABADOS_TREINO, DiaForaDoDesenho, tipo_do_dia)


def categorias(dia):
    """Todas as categorias em que o dia cai (o certo é exatamente uma)."""
    semana = datetime.date(2016, 3, dia).weekday()
    listas = {"treino": DIAS_TREINO, "sabado_treino": SABADOS_TREINO, "validacao": DIAS_VALIDACAO,
              "avaliacao": DIAS_AVALIACAO, "extra": DIAS_AVALIACAO_EXTRAS, "excluido": DIAS_EXCLUIDOS}
    achadas = [nome for nome, dias in listas.items() if dia in dias]
    if semana == 4:
        achadas.append("sexta")
    if semana == 6:
        achadas.append("domingo")
    return achadas


@pytest.mark.parametrize("dia", range(1, 32))
def test_todo_dia_tem_exatamente_uma_categoria(dia):
    assert len(categorias(dia)) == 1, (dia, categorias(dia))


def test_dias_com_dono_tem_o_tipo_certo():
    for dia in DIAS_TREINO + DIAS_VALIDACAO + DIAS_AVALIACAO_EXTRAS + [23]:
        assert tipo_do_dia(dia) == "util", dia
    for dia in SABADOS_TREINO + [19]:
        assert tipo_do_dia(dia) == "sabado", dia


@pytest.mark.parametrize("dia", [17, 26, 18, 25, 20, 27])
def test_excluidos_sextas_e_domingos_dao_erro(dia):
    with pytest.raises(DiaForaDoDesenho):
        tipo_do_dia(dia)
