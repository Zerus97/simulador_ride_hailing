"""Gabarito: a mesma dinâmica do simulador, em tempo contínuo, de evento em evento.

O simulador avança de 10 em 10 minutos; o gabarito pula para o próximo evento, em
ordem de horário (uma fila de prioridade, heapq). Dois tipos de evento:
  - um pedido chega, no segundo registrado;
  - um carro fica livre, no segundo exato: início + duração × 60.

As regras são as mesmas do simulador, mas o código é independente de propósito
(só heapq e deque): a prova vem de dois códigos diferentes chegarem ao mesmo
número, e um erro compartilhado pelos dois passaria despercebido.
"""
import heapq
from collections import deque

SEGUNDOS_POR_PASSO = 600  # repetido aqui de propósito: nada é importado do simulador
CARRO, PEDIDO = 0, 1      # num empate de horário, o carro livre é processado antes do pedido


def taxa_gabarito(pedidos, zonas_iniciais, fim_segundos):
    """Fração de pedidos atendidos (atendidos ÷ total) em tempo contínuo.

    pedidos: os pedidos da simulação (sim.historico_pedidos), com origem, destino,
        segundo_do_dia e duracao_min.
    zonas_iniciais: a zona de cada carro antes do primeiro passo.
    fim_segundos: o último matching do simulador, (passo_inicial + passos) * 600;
        nenhum atendimento depois disso conta."""
    if not pedidos:
        return float("nan")

    eventos = []
    contador = 0  # único por evento: evita comparar dois Pedido num empate de (tempo, tipo)

    def agendar(tempo, tipo, dados):
        nonlocal contador
        heapq.heappush(eventos, (tempo, tipo, contador, dados))
        contador += 1

    ociosos = {}  # zona -> carros parados nela
    for zona in zonas_iniciais:
        ociosos[zona] = ociosos.get(zona, 0) + 1
    fila = {}     # zona -> deque de (prazo, pedido), do mais antigo ao mais novo
    for pedido in pedidos:
        agendar(pedido.segundo_do_dia, PEDIDO, pedido)

    atendidos = 0

    def atender(tempo, pedido):
        nonlocal atendidos
        atendidos += 1
        agendar(tempo + pedido.duracao_min * 60, CARRO, pedido.destino)

    while eventos:
        tempo, tipo, _, dados = heapq.heappop(eventos)
        if tempo > fim_segundos:
            break  # tudo o que sobra é depois do último matching

        if tipo == PEDIDO:
            pedido = dados
            zona = pedido.origem
            if ociosos.get(zona, 0) > 0:
                ociosos[zona] -= 1
                atender(tempo, pedido)
            else:
                # o mesmo prazo do expira = criado_em + 2 do simulador: o último
                # atendimento possível é no fim da janela seguinte à do pedido
                prazo = (pedido.segundo_do_dia // SEGUNDOS_POR_PASSO + 2) * SEGUNDOS_POR_PASSO
                fila.setdefault(zona, deque()).append((prazo, pedido))

        else:  # um carro fica livre na zona `dados`
            zona = dados
            espera = fila.get(zona)
            while espera and espera[0][0] < tempo:
                espera.popleft()  # prazo vencido: o pedido expirou
            if espera:
                _, pedido = espera.popleft()
                atender(tempo, pedido)
            else:
                ociosos[zona] = ociosos.get(zona, 0) + 1

    return atendidos / len(pedidos)
