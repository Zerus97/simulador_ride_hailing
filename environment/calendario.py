"""O calendário de março/2016 do desenho experimental (Q3): o papel de cada dia.

Fica no environment/, a camada leve (só a biblioteca padrão), para que o ETL, o
simulador e o executor de experimentos leiam as mesmas listas sem puxar as
bibliotecas de geografia. Todo dia de 1 a 31 cai em exatamente uma categoria:
treino, sábado de treino, validação, avaliação, avaliação extra, excluído, ou fora
por ser sexta ou domingo (tests/test_calendario.py confere).
"""
import datetime

DIAS_TREINO = [1, 2, 3, 7, 8, 9, 10]        # dias úteis: λ, fatores, preferências, pontos, grade
SABADOS_TREINO = [5, 12]                    # o cenário de fim de semana é o sábado 19/mar
DIAS_VALIDACAO = [14, 15, 16]               # seg–qua: validação fora da amostra dos fatores
DIAS_AVALIACAO = [23, 19]                   # qua 23/mar (dia útil) e sáb 19/mar (fim de semana)
DIAS_AVALIACAO_EXTRAS = [21, 22, 24, 28, 29, 30, 31]  # reservados para avaliação extra
# 17/mar (quinta) e o fim de semana da Páscoa (26–27/mar). O 27 já cai fora por ser
# domingo; o 26 é sábado e precisa estar aqui.
DIAS_EXCLUIDOS = [17, 26]

ANO, MES = 2016, 3


class DiaForaDoDesenho(ValueError):
    """Dia sem lugar nos experimentos: sexta, domingo ou excluído."""


def tipo_do_dia(dia):
    """ "util" (segunda a quinta) ou "sabado". Sexta e domingo ficaram fora do desenho
    (Q3) e não têm previsão; os DIAS_EXCLUIDOS também."""
    if dia in DIAS_EXCLUIDOS:
        raise DiaForaDoDesenho(f"o dia {dia}/mar foi excluído do desenho")
    dia_semana = datetime.date(ANO, MES, dia).weekday()
    if dia_semana <= 3:
        return "util"
    if dia_semana == 5:
        return "sabado"
    nome = "sexta" if dia_semana == 4 else "domingo"
    raise DiaForaDoDesenho(f"o dia {dia}/mar é {nome}: fora do desenho (Q3), sem previsão")
