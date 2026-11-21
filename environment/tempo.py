"""O relógio do simulador: passos de 10 minutos.

Fica no environment/, a camada de baixo, para que todos possam importar daqui: o
simulador, as fontes de demanda e o ETL.

Convenção do relógio: tudo o que acontece num passo (matching, recomendação) acontece
no FIM da janela de 10 minutos. Um pedido das 3h07 está no passo do dia 18 (3h00 a
3h10); atendido nesse mesmo passo, ele esperou até as 3h10, ou seja, 3 minutos.
"""
import math

SEGUNDOS_POR_PASSO = 600                          # passos de 10 min
MINUTOS_POR_PASSO = SEGUNDOS_POR_PASSO // 60      # 10
PASSOS_POR_DIA = 24 * 3600 // SEGUNDOS_POR_PASSO  # 144


def passos_da_corrida(duracao_min):
    """Passos que o carro fica ocupado: o mais próximo, com mínimo de 1.
    floor(x + 0,5), e não round(): o round() do Python leva 25 min para 2 passos."""
    if duracao_min is None:          # pedidos da Poisson: 1 passo, como hoje
        return 1
    return max(1, math.floor(duracao_min / MINUTOS_POR_PASSO + 0.5))


def espera_min(segundo_do_dia, passos_ate_atender):
    """Minutos entre o pedido e o fim do passo em que ele foi atendido.

    passos_ate_atender: quantos passos ele esperou (tempo do atendimento - criado_em).
    Com o atendimento no fim da janela, a espera vai de quase 0 a 20 min (w = 1)."""
    passo_atendimento = segundo_do_dia // SEGUNDOS_POR_PASSO + passos_ate_atender
    return ((passo_atendimento + 1) * SEGUNDOS_POR_PASSO - segundo_do_dia) / 60
