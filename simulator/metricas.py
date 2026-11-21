"""Métricas da proposta (Tab. 3.4), sem a hora de aquecimento.

O metrics() do Simulador continua com as métricas antigas (a regressão depende dele).
Este coletor recebe os eventos de cada passo (Simulador.passo chama registrar) e
calcula as métricas da proposta. Não sorteia nada.

Aquecimento: nada do que acontece nos primeiros `passos_aquecimento` passos entra nas
métricas. Os pedidos criados no aquecimento ficam fora do numerador e do denominador
do atendimento; os movimentos do aquecimento ficam fora do esforço.

Quando uma informação não existe (pedidos da Poisson não têm km nem horário; sem
tempos não há minutos nem km de reposicionamento), a métrica sai None, e não 0:
"não se aplica" não pode ser confundido com "deu zero".
"""
import math

TARIFA_USD_KM = 3.25          # Q4
TIPOS_MOVIMENTO = ("recomendado", "pos_recusa", "proprio")


class ColetorMetricas:
    def __init__(self, passos_aquecimento, tarifa_km=TARIFA_USD_KM, tempos=None):
        self.passos_aquecimento = passos_aquecimento
        self.tarifa_km = tarifa_km
        self.tempos = tempos
        self.pedidos_gerados = 0
        self.corridas = []      # corridas de pedidos criados depois do aquecimento
        self.movimentos = []    # (id, origem, destino, tipo, hora) depois do aquecimento

    def fora_do_aquecimento(self, tempo):
        return tempo > self.passos_aquecimento

    def registrar(self, relatorio, hora, pedidos_novos, corridas_novas):
        """Eventos do passo relatorio.tempo: o relatório, a hora do dia, os pedidos
        criados e as corridas iniciadas (o matching) nesse passo."""
        self.pedidos_gerados += sum(self.fora_do_aquecimento(p.criado_em) for p in pedidos_novos)
        self.corridas.extend(c for c in corridas_novas if self.fora_do_aquecimento(c.pedido.criado_em))
        if self.fora_do_aquecimento(relatorio.tempo):
            self.movimentos.extend((*m, hora) for m in relatorio.movimentos)

    def resultado(self):
        atendidos = len(self.corridas)
        esperas = [c.espera_min for c in self.corridas]
        kms = [c.pedido.km for c in self.corridas]
        recomendados = [m for m in self.movimentos if m[3] == "recomendado"]
        return {
            "pedidos_gerados": self.pedidos_gerados,
            "pedidos_atendidos": atendidos,
            "atendimento": atendidos / self.pedidos_gerados if self.pedidos_gerados else math.nan,
            "espera_media_min": _media_ou_none(esperas),
            "receita_usd": (self.tarifa_km * sum(kms)
                            if kms and all(k is not None for k in kms) else None),
            "reposicionamentos": len(recomendados),
            "reposicionamento_min": (sum(self.tempos.minutos(o, d, h) for _, o, d, _, h in recomendados)
                                     if self.tempos is not None else None),
            "reposicionamento_km": (sum(self.tempos.km(o, d) for _, o, d, _, _ in recomendados)
                                    if self.tempos is not None else None),
        }


def _media_ou_none(valores):
    if not valores or any(v is None for v in valores):
        return None
    return sum(valores) / len(valores)
