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


LIMIAR_ARREDONDAMENTO = 0.0   # c em floor(min/10 + c); calibrado nos dias úteis de treino (21/nov)


def passos_da_corrida(duracao_min):
    """Passos que o carro fica ocupado: para baixo, com mínimo de 1.
    O mínimo infla as corridas curtas; arredondar para baixo compensa no total.

    O limiar é lido aqui dentro, a cada chamada, para que um teste possa trocá-lo
    (monkeypatch) e comparar as duas versões."""
    if duracao_min is None:          # pedidos da Poisson: 1 passo, como hoje
        return 1
    return max(1, math.floor(duracao_min / MINUTOS_POR_PASSO + LIMIAR_ARREDONDAMENTO))


def espera_min(segundo_do_dia, passos_ate_atender):
    """Minutos entre o pedido e o fim do passo em que ele foi atendido.

    passos_ate_atender: quantos passos ele esperou (tempo do atendimento - criado_em).
    Com o atendimento no fim da janela, a espera vai de quase 0 a 20 min (w = 1)."""
    passo_atendimento = segundo_do_dia // SEGUNDOS_POR_PASSO + passos_ate_atender
    return ((passo_atendimento + 1) * SEGUNDOS_POR_PASSO - segundo_do_dia) / 60
