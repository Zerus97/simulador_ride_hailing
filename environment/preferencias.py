"""Preferências dos motoristas entre ficar e as vizinhas (fechamento da Q7, tarefa 1.8).

Para um motorista com zona-base b, na zona atual z, na hora h, a preferência por cada
candidata c (a própria z ou uma vizinha) é proporcional a

    D(c, h) ^ peso_demanda × exp(−α · d(c, b))

  - D(c, h) = λ da hora (soma dos 6 passos, na escala de todos os dados) + 0,5 / n_dias:
    a correção de meio pedido nos dias de treino evita preferência zero onde não houve
    pedido (1 pedido em n dias já fica na frente de nenhum);
  - d(c, b) = distância em km entre os centros das células;
  - α equilibra os dois termos, cada um ponderado por onde os motoristas estão:
    média do desvio-padrão de log D entre as candidatas, com cada (zona atual, hora)
    pesada pela demanda daquela hora ÷ média do desvio-padrão de d(·, base) entre as
    candidatas, com cada (zona atual, base) pesada por demanda da zona × demanda da base.
    Sem a ponderação, as zonas de borda (muitos zeros, pouca demanda) dominavam o
    numerador e inflavam o α (1,81 contra 1,47 na grade de 1,2 km, dias úteis).
    O α é o que sai da conta, sem ajuste pelo comportamento que ele produz.

Não há sorteio aqui: dados a base, a zona atual e a hora, o resultado é sempre o mesmo.
Quem sorteia pelas preferências é o módulo de comportamento (tarefa 1.9).
Fica no environment/, a camada leve: só a biblioteca padrão.
"""
import math
from statistics import fmean, pstdev

from environment.calendario import DIAS_TREINO, SABADOS_TREINO
from environment.previsao import previsao_do_csv

HORAS_POR_DIA = 24
PASSOS_POR_HORA = 6
# Dias de treino por tipo de dia, das próprias listas do calendário.
DIAS_POR_TIPO = {"util": len(DIAS_TREINO), "sabado": len(SABADOS_TREINO)}
CORRECAO_PEDIDOS = 0.5  # meio pedido no total dos dias de treino


class Preferencias:
    """Preferência de cada motorista entre a zona atual e as vizinhas, por hora."""

    def __init__(self, grade, lado, lam_hora, demanda_hora, pesos_base, peso_demanda=1.0):
        """lam_hora: (zona, hora) -> λ da hora, sem correção (os pesos do numerador do α).
        demanda_hora: (zona, hora) -> D (já com a correção de 0,5 / n_dias).
        pesos_base: zona -> probabilidade de ser sorteada como base (a demanda do dia).
        peso_demanda: 1,0 no modelo; 0,0 é a variação sem demanda (só a distância)."""
        self.grade = grade
        self.lado = lado
        self.lam_hora = dict(lam_hora)
        self.demanda_hora = dict(demanda_hora)
        self.pesos_base = dict(pesos_base)
        self.peso_demanda = peso_demanda
        self._cache = {}
        self.alfa = self._calcular_alfa()

    def candidatas(self, zona):
        return [zona] + list(self.grade.vizinhas(zona))

    def distancia(self, a, b):
        """km entre os centros das células: lado/1000 × √(Δlinha² + Δcoluna²)."""
        (la, ca), (lb, cb) = self.grade.celula(a), self.grade.celula(b)
        return self.lado / 1000 * math.hypot(la - lb, ca - cb)

    def _calcular_alfa(self):
        zonas = self.grade.zonas()
        # numerador: cada (zona atual, hora) pesa a demanda daquela hora
        termos, pesos = [], []
        for z in zonas:
            for h in range(HORAS_POR_DIA):
                if self.lam_hora[(z, h)] > 0:
                    termos.append(pstdev([math.log(self.demanda_hora[(c, h)]) for c in self.candidatas(z)]))
                    pesos.append(self.lam_hora[(z, h)])
        if not pesos:
            raise ValueError("nenhuma demanda: α indefinido")
        numerador = sum(t * p for t, p in zip(termos, pesos)) / sum(pesos)
        # denominador: cada (zona atual, base) pesa demanda da zona × demanda da base
        termos, pesos = [], []
        for z in zonas:
            for b, wb in self.pesos_base.items():
                if self.pesos_base[z] > 0 and wb > 0:
                    termos.append(pstdev([self.distancia(c, b) for c in self.candidatas(z)]))
                    pesos.append(self.pesos_base[z] * wb)
        denominador = sum(t * p for t, p in zip(termos, pesos)) / sum(pesos)
        if denominador == 0:
            raise ValueError("as distâncias às bases não variam entre as candidatas: α indefinido")
        return numerador / denominador

    def de(self, base, zona_atual, hora):
        """{zona: probabilidade} para a zona atual e as vizinhas (soma 1).
        hora 24 ou mais (última decisão do dia) usa a do dia seguinte."""
        chave = (base, zona_atual, hora % HORAS_POR_DIA)
        if chave not in self._cache:
            h = chave[2]
            candidatas = self.candidatas(zona_atual)
            # em log, para não estourar: peso_demanda · log D − α · d
            logs = [self.peso_demanda * math.log(self.demanda_hora[(c, h)])
                    - self.alfa * self.distancia(c, base) for c in candidatas]
            maior = max(logs)
            pesos = [math.exp(x - maior) for x in logs]
            total = sum(pesos)
            self._cache[chave] = {c: p / total for c, p in zip(candidatas, pesos)}
        return self._cache[chave]


def preferencias_do_csv(caminho_lambda, grade, lado, tipo_dia, n_dias, peso_demanda=1.0):
    """Preferencias a partir do lambda_{lado}.csv (a escala de todos os dados), para um tipo
    de dia; n_dias = número de dias de treino desse tipo (DIAS_POR_TIPO). Imprime o α."""
    previsao = previsao_do_csv(caminho_lambda, grade, tipo_dia, fracao=1.0)
    lam_hora, demanda_hora, demanda_dia = {}, {}, {}
    for z in grade.zonas():
        for h in range(HORAS_POR_DIA):
            lam = sum(previsao.lam(z, PASSOS_POR_HORA * h + k) for k in range(PASSOS_POR_HORA))
            lam_hora[(z, h)] = lam
            demanda_hora[(z, h)] = lam + CORRECAO_PEDIDOS / n_dias
            demanda_dia[z] = demanda_dia.get(z, 0.0) + lam
    total = sum(demanda_dia.values())
    pesos_base = {z: d / total for z, d in demanda_dia.items()}
    pref = Preferencias(grade, lado, lam_hora, demanda_hora, pesos_base, peso_demanda)
    print(f"preferências {lado} m, {tipo_dia}: α = {pref.alfa:.4f} por km")
    return pref
