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

from environment.env import EM_CORRIDA, OCIOSO, REPOSICIONANDO

TARIFA_USD_KM = 3.25          # Q4
CUSTO_USD_KM = 0.23           # custo de rodar, em todo movimento (os três tipos)
TIPOS_MOVIMENTO = ("recomendado", "pos_recusa", "proprio")


class ColetorMetricas:
    def __init__(self, passos_aquecimento, tarifa_km=TARIFA_USD_KM, tempos=None):
        self.passos_aquecimento = passos_aquecimento
        self.tarifa_km = tarifa_km
        self.tempos = tempos
        self.pedidos_gerados = 0
        self.corridas = []      # corridas de pedidos criados depois do aquecimento
        self.movimentos = []    # (id, origem, destino, tipo, hora) depois do aquecimento
        self.aceites = self.recusas = 0
        self.sucessos = self.avaliados = 0
        self._recomendados_anteriores = []   # (id, destino) do passo anterior, à espera do matching
        self.estados = {}       # estado -> soma de veículos-passo
        self.num_veiculos = None
        self.tempos_decisao = []
        self.confianca_final = None   # confiança média da frota no último passo registrado

    def fora_do_aquecimento(self, tempo):
        return tempo > self.passos_aquecimento

    def registrar(self, relatorio, hora, pedidos_novos, corridas_novas, estados=None, tempo_decisao=None,
                  confianca=None):
        """Eventos do passo relatorio.tempo: o relatório, a hora do dia, os pedidos
        criados e as corridas iniciadas (o matching) nesse passo, quantos veículos há em
        cada estado no fim do passo e o tempo de relógio do decidir (s)."""
        self.pedidos_gerados += sum(self.fora_do_aquecimento(p.criado_em) for p in pedidos_novos)
        self.corridas.extend(c for c in corridas_novas if self.fora_do_aquecimento(c.pedido.criado_em))

        # sucesso: um "recomendado" de t é sucesso se o veículo aparece nos casamentos de
        # t + 1 (no matching de t + 1, ele já está na zona de destino). Um movimento do
        # último passo não tem o t + 1 e fica fora da conta.
        casados = {(vid, zona) for vid, _, zona in relatorio.casamentos}
        for vid, destino in self._recomendados_anteriores:
            self.avaliados += 1
            self.sucessos += (vid, destino) in casados
        self._recomendados_anteriores = []

        if estados is not None:
            self.num_veiculos = sum(estados.values())
        self.confianca_final = confianca  # "no fim do dia": o valor do último passo
        if not self.fora_do_aquecimento(relatorio.tempo):
            return
        self.movimentos.extend((*m, hora) for m in relatorio.movimentos)
        self.aceites += len(relatorio.aceites)
        self.recusas += len(relatorio.recusas)
        self._recomendados_anteriores = [(vid, d) for vid, _, d, tipo in relatorio.movimentos
                                         if tipo == "recomendado"]
        if estados is not None:
            for estado, n in estados.items():
                self.estados[estado] = self.estados.get(estado, 0) + n
        if tempo_decisao is not None:
            self.tempos_decisao.append(tempo_decisao)

    def resultado(self):
        atendidos = len(self.corridas)
        esperas = [c.espera_min for c in self.corridas]
        kms = [c.pedido.km for c in self.corridas]
        recomendados = [m for m in self.movimentos if m[3] == "recomendado"]
        receita = (self.tarifa_km * sum(kms) if kms and all(k is not None for k in kms) else None)
        km_todos = (sum(self.tempos.km(o, d) for _, o, d, _, _ in self.movimentos)
                    if self.tempos is not None else None)
        total_estados = sum(self.estados.values())
        return {
            "pedidos_gerados": self.pedidos_gerados,
            "pedidos_atendidos": atendidos,
            "atendimento": atendidos / self.pedidos_gerados if self.pedidos_gerados else math.nan,
            "espera_media_min": _media_ou_none(esperas),
            "receita_usd": receita,
            "reposicionamentos": len(recomendados),
            "reposicionamento_min": (sum(self.tempos.minutos(o, d, h) for _, o, d, _, h in recomendados)
                                     if self.tempos is not None else None),
            "reposicionamento_km": (sum(self.tempos.km(o, d) for _, o, d, _, _ in recomendados)
                                    if self.tempos is not None else None),
            # --- secundárias ---
            "aceite": _razao(self.aceites, self.aceites + self.recusas),
            "reposicionamentos_avaliados": self.avaliados,
            "reposicionamento_sucesso": _razao(self.sucessos, self.avaliados),
            "ganhos_usd_por_veiculo": ((receita - CUSTO_USD_KM * km_todos) / self.num_veiculos
                                       if None not in (receita, km_todos, self.num_veiculos) else None),
            "frac_parado": _razao(self.estados.get(OCIOSO, 0), total_estados),
            "frac_reposicionando": _razao(self.estados.get(REPOSICIONANDO, 0), total_estados),
            "frac_em_corrida": _razao(self.estados.get(EM_CORRIDA, 0), total_estados),
            "decisao_media_s": (sum(self.tempos_decisao) / len(self.tempos_decisao)
                                if self.tempos_decisao else None),
            "decisao_max_s": max(self.tempos_decisao) if self.tempos_decisao else None,
            "confianca_frota": self.confianca_final,  # None se o modelo de aceite não evolui
        }


def _razao(a, b):
    return a / b if b else math.nan


def _media_ou_none(valores):
    if not valores or any(v is None for v in valores):
        return None
    return sum(valores) / len(valores)
