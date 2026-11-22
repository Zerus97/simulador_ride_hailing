"""Comportamento dos motoristas diante da recomendação (o A da proposta, tarefa 1.9).

Fica no environment/: o comportamento é do ambiente, não do algoritmo. Três peças:
  - ModeloAceite: a probabilidade de aceitar a recomendação (sem sortear);
  - RegraPosRecusa: para onde vai quem recusou;
  - Comportamento: junta as duas e responde um veículo por vez, a partir de dois
    números uniformes que o simulador sorteia SEMPRE (u_aceite e u_destino), usados ou
    não. Assim a quantidade de sorteios não depende do que o modelo decide, e o aceite
    fica encaixado entre valores de ρ (quem aceita com 0,3 também aceita com 0,7).
"""
import math
from dataclasses import dataclass

import numpy as np

SEM_RECOMENDACAO = ("ficar", "preferencias")


@dataclass(frozen=True)
class Contexto:
    """O que o ambiente sabe do motorista na hora da resposta."""
    hora: int              # passo_do_dia // 6 (o resto por 24 fica com Preferencias.de)
    preferencias: object   # environment.preferencias.Preferencias, ou None
    grade: object          # o grafo de zonas
    tipo_dia: str = None   # "util"/"sabado" (None: sem calendário, ex.: Poisson; vale como útil)
    tempo: int = None      # passo atual da simulação (para a espera do motorista)
    tempos: object = None  # environment.tempos.TemposEntreZonas, ou None
    previsao: object = None      # λ (environment.previsao), para a logística do i-Rebalance
    passo_do_dia: int = None
    renda: object = None         # environment.renda.RendaEsperada
    oferta: dict = None          # zona -> carros: ociosos + chegadas em t+1, atualizada a cada movimento


def candidatas(zona, grade):
    """A zona atual e as vizinhas, numa ordem fixa: a atual, depois a de grade.vizinhas."""
    return [zona] + list(grade.vizinhas(zona))


def pela_acumulada(zonas, probs, u):
    """A primeira zona em que a probabilidade acumulada passa de u (u em [0, 1))."""
    acumulada = 0.0
    for zona, p in zip(zonas, probs):
        acumulada += p
        if u < acumulada:
            return zona
    return zonas[-1]  # arredondamento: a soma pode dar 0,999...


def por_preferencia(veiculo, ctx):
    """As candidatas da maior para a menor preferência; o empate fica com a ordem das
    candidatas (a zona atual primeiro, depois a de grade.vizinhas), porque sorted é estável."""
    pref = ctx.preferencias.de(veiculo.base, veiculo.zona, ctx.hora)
    return sorted(candidatas(veiculo.zona, ctx.grade), key=lambda z: -pref[z])


def pelas_preferencias(veiculo, ctx, u):
    """Sorteia entre a zona atual e as vizinhas pelas preferências do motorista na hora."""
    pref = ctx.preferencias.de(veiculo.base, veiculo.zona, ctx.hora)
    zonas = candidatas(veiculo.zona, ctx.grade)
    return pela_acumulada(zonas, [pref[z] for z in zonas], u)


# --- modelos de aceite ---

class ModeloAceite:
    def prob(self, veiculo, zona_recomendada, ctx):
        """Probabilidade de o motorista aceitar a recomendação (sem sortear)."""
        raise NotImplementedError


class AceiteFixo(ModeloAceite):
    """Todo motorista aceita com a mesma probabilidade ρ."""

    def __init__(self, rho):
        if not 0 <= rho <= 1:
            raise ValueError(f"ρ precisa estar entre 0 e 1: {rho}")
        self.rho = rho

    def prob(self, veiculo, zona_recomendada, ctx):
        return self.rho


# Distribuição de obediência dos motoristas: questionaire.csv do repositório do
# i-Rebalance (licença MIT), coluna obey, contada por motorista (87 motoristas, um valor
# cada; 1.854 respostas no arquivo). Média: 0,490.
OBEDIENCIA_NIVEIS = (0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.45, 0.5, 0.51, 0.6, 0.7, 0.8, 0.9, 1.0)
OBEDIENCIA_CONTAGENS = (6, 5, 11, 5, 6, 1, 1, 14, 1, 11, 6, 9, 5, 6)


class ConfiancaQuestionario(ModeloAceite):
    """Cada motorista aceita com a sua própria obediência, sorteada uma vez da
    distribuição do questionário e fixa durante toda a simulação.

    A obediência fica aqui no modelo, e não no Veiculo: é o A do ambiente, e assim não
    vaza para os algoritmos pela observação (o AAVR terá o seu μ̂ separado)."""

    def __init__(self, niveis=OBEDIENCIA_NIVEIS, contagens=OBEDIENCIA_CONTAGENS):
        if len(niveis) != len(contagens) or any(c < 0 for c in contagens) or sum(contagens) <= 0:
            raise ValueError("a tabela de obediência precisa de níveis e contagens válidos")
        if any(not 0 <= n <= 1 for n in niveis):
            raise ValueError("os níveis de obediência precisam estar entre 0 e 1")
        self.niveis = list(niveis)
        self.contagens = list(contagens)
        self.obediencia = None

    def iniciar(self, veiculos, rng):
        """Sorteia uma obediência por veículo, em ordem de id, num único choice."""
        p = np.asarray(self.contagens, dtype=float) / sum(self.contagens)
        sorteio = rng.choice(self.niveis, size=len(veiculos), p=p)
        self.obediencia = {v.id: float(o) for v, o in zip(veiculos, sorteio)}

    def prob(self, veiculo, zona_recomendada, ctx):
        if self.obediencia is None:
            raise RuntimeError("ConfiancaQuestionario sem iniciar(): o simulador sorteia as obediências")
        return self.obediencia[veiculo.id]


class ConfiancaEvolutiva(ConfiancaQuestionario):
    """Confiança evolutiva (atualização Beta do AAVR): cada motorista começa com a
    obediência do questionário, μ₀ (o mesmo sorteio, gerador e ordem da
    ConfiancaQuestionario), como um Beta(α = μ₀ · n₀, β = (1 − μ₀) · n₀), e aceita com
    probabilidade α ÷ (α + β).

    n₀ é a força inicial: n₀ = 48 é "um dia de experiência", a média de aceites por
    motorista num dia útil (dia 23, 1.500 carros, gulosa recomendando a todos os ociosos,
    contando também as recomendações de ficar).

    Atualização (regra da Q9), no fim de cada passo, inclusive no aquecimento (é a
    dinâmica do ambiente): quem aceitou no passo anterior, mover ou ficar, ganha α += 1
    se aparece nos casamentos deste passo, e β += 1 se não. Quem recusa ou não recebe
    recomendação não atualiza; por isso μ₀ = 0 nunca aceita nem muda, e μ₀ = 1 aceita até
    o primeiro fracasso."""

    def __init__(self, n0, niveis=OBEDIENCIA_NIVEIS, contagens=OBEDIENCIA_CONTAGENS):
        super().__init__(niveis, contagens)
        if not n0 > 0:
            raise ValueError(f"n₀ precisa ser maior que 0: {n0}")
        self.n0 = n0
        self.alfa = self.beta = None
        self._aceites_anteriores = ()

    def iniciar(self, veiculos, rng):
        super().iniciar(veiculos, rng)
        self.alfa = {vid: mu * self.n0 for vid, mu in self.obediencia.items()}
        self.beta = {vid: (1 - mu) * self.n0 for vid, mu in self.obediencia.items()}
        self._aceites_anteriores = ()

    def confianca(self, vid):
        return self.alfa[vid] / (self.alfa[vid] + self.beta[vid])

    def prob(self, veiculo, zona_recomendada, ctx):
        if self.alfa is None:
            raise RuntimeError("ConfiancaEvolutiva sem iniciar(): o simulador sorteia as obediências")
        return self.confianca(veiculo.id)

    def fim_do_passo(self, relatorio):
        """Confere os aceites do passo anterior contra os casamentos deste passo e guarda
        os aceites deste passo para o próximo."""
        casados = {vid for vid, _, _ in relatorio.casamentos}
        for vid in self._aceites_anteriores:
            if vid in casados:
                self.alfa[vid] += 1
            else:
                self.beta[vid] += 1
        self._aceites_anteriores = tuple(relatorio.aceites)

    def confianca_media(self):
        """A confiança da frota: a média de α ÷ (α + β) entre os motoristas."""
        return sum(self.confianca(vid) for vid in self.alfa) / len(self.alfa)


# Coeficientes da logística do i-Rebalance, (b, w_r, w_m, w_o):
#   P(aceitar) = σ(b + w_r · r + w_m · m + w_o · o)
# "publicados" (padrão): Eq. 1 e tabela de Chen et al. (2024).
# "codigo": a regra do repositório deles (só na validação da Fase 4).
# "reajuste": a logística reajustada nas 1.854 respostas do questionaire.csv (aceite =
#   1 − UnAccepted; r = "Ranking ", m = Revenue, o = obey contínuo); erros-padrão 0,254;
#   0,025; 0,022; 0,210; aceitação média no questionário 0,662.
COEFICIENTES_LOGISTICA = {
    "publicados": (-1.31, -0.44, 0.29, 2.17),
    "codigo": (-1.3959, -0.3892, 0.2932, 1.8379),
    "reajuste": (-1.306, -0.403, 0.291, 1.858),
}
# m = 14,6 × fator(z) × D ÷ max(S, D): 14,6 RMB é a renda média por corrida declarada
# pelos motoristas no questionário (10 a 20 RMB), a âncora da conversão (Q10). Sem limite
# em [6, 16].
RENDA_REFERENCIA = 14.6


class AceiteRecomendacao(ConfiancaQuestionario):
    """Logística do i-Rebalance: o aceite depende da posição da recomendada entre as
    preferidas (r), da renda esperada nela (m) e da obediência do motorista (o, o
    sorteio da ConfiancaQuestionario).

      r = posição da recomendada entre as candidatas pelas preferências (1 = favorita;
          empate pela ordem das candidatas);
      m = 14,6 × fator(z) × D ÷ max(S, D), com D = λ(z, passo seguinte) e S a oferta em z
          sem o próprio motorista; m = 0 se D = 0. A oferta é atualizada a cada movimento
          no mesmo passo, então quem decide depois vê a aglomeração dos anteriores.

    prob = σ(logit); no modo determinístico, 1 se logit ≥ 0 e 0 se não (a regra do código
    deles). O sorteio com u_aceite continua igual, e o contrato não muda."""
    precisa_preferencias = precisa_previsao = precisa_renda = True

    def __init__(self, deterministico=False, coeficientes="publicados",
                 niveis=OBEDIENCIA_NIVEIS, contagens=OBEDIENCIA_CONTAGENS):
        super().__init__(niveis, contagens)
        if isinstance(coeficientes, str):
            if coeficientes not in COEFICIENTES_LOGISTICA:
                raise ValueError(f"coeficientes {coeficientes!r} ainda não registrados; "
                                 f"disponíveis: {sorted(COEFICIENTES_LOGISTICA)}")
            coeficientes = COEFICIENTES_LOGISTICA[coeficientes]
        self.b, self.w_r, self.w_m, self.w_o = coeficientes
        self.deterministico = deterministico

    def posicao(self, veiculo, zona_recomendada, ctx):
        return por_preferencia(veiculo, ctx).index(zona_recomendada) + 1

    def m(self, veiculo, zona_recomendada, ctx):
        demanda = ctx.previsao.lam(zona_recomendada, ctx.passo_do_dia + 1)
        if demanda == 0:
            return 0.0
        oferta = ctx.oferta.get(zona_recomendada, 0) - (1 if zona_recomendada == veiculo.zona else 0)
        return RENDA_REFERENCIA * ctx.renda.fator(zona_recomendada) * demanda / max(oferta, demanda)

    def logit_de(self, r, m, o):
        return self.b + self.w_r * r + self.w_m * m + self.w_o * o

    def prob_de(self, r, m, o):
        """σ(logit), ou, no modo determinístico, 1 se logit ≥ 0 e 0 se não."""
        x = self.logit_de(r, m, o)
        if self.deterministico:
            return 1.0 if x >= 0 else 0.0
        return 1 / (1 + math.exp(-x))

    def variaveis(self, veiculo, zona_recomendada, ctx):
        """(r, m, o) do motorista para a zona recomendada."""
        return (self.posicao(veiculo, zona_recomendada, ctx), self.m(veiculo, zona_recomendada, ctx),
                super().prob(veiculo, zona_recomendada, ctx))

    def logit(self, veiculo, zona_recomendada, ctx):
        return self.logit_de(*self.variaveis(veiculo, zona_recomendada, ctx))

    def prob(self, veiculo, zona_recomendada, ctx):
        return self.prob_de(*self.variaveis(veiculo, zona_recomendada, ctx))


# --- regras pós-recusa ---

class RegraPosRecusa:
    precisa_preferencias = False

    def destino(self, veiculo, zona_recomendada, ctx, u):
        """Zona para onde vai quem recusou (a atual ou uma vizinha)."""
        raise NotImplementedError


class Ficar(RegraPosRecusa):
    """Quem recusa fica onde está."""

    def destino(self, veiculo, zona_recomendada, ctx, u):
        return veiculo.zona


class Proporcional(RegraPosRecusa):
    """Quem recusa sorteia entre a zona atual e as vizinhas pelas suas preferências;
    a recomendada não é excluída (como no AAVR)."""
    precisa_preferencias = True

    def destino(self, veiculo, zona_recomendada, ctx, u):
        return pelas_preferencias(veiculo, ctx, u)


class ZonaPreferida(RegraPosRecusa):
    """Quem recusa vai para a candidata de maior preferência (a zona atual ou uma
    vizinha); empate pela ordem das candidatas. Não sorteia: o u não é usado, mas o
    simulador continua tirando os dois números, para não mudar o contrato."""
    precisa_preferencias = True

    def destino(self, veiculo, zona_recomendada, ctx, u):
        return por_preferencia(veiculo, ctx)[0]


class UniformeQuatroPreferidas(RegraPosRecusa):
    """Quem recusa sorteia por igual entre as 4 candidatas de maior preferência (todas,
    se houver menos de 4). A zona atual e a recomendada podem estar entre elas, como no
    i-Rebalance. As 4 são ordenadas antes do sorteio (preferência, depois a ordem das
    candidatas), o que torna o resultado reprodutível: índice = int(u × quantas)."""
    precisa_preferencias = True
    QUANTAS = 4

    def destino(self, veiculo, zona_recomendada, ctx, u):
        preferidas = por_preferencia(veiculo, ctx)[:self.QUANTAS]
        return preferidas[int(u * len(preferidas))]


# --- o comportamento ---

class Comportamento:
    """sem_recomendacao: o que faz o motorista que não recebeu recomendação.
      "ficar": fica (baseline "sem reposicionamento");
      "preferencias": sorteia pelas preferências (baseline "sem recomendação", Q8)."""

    def __init__(self, modelo, regra, sem_recomendacao="ficar"):
        if sem_recomendacao not in SEM_RECOMENDACAO:
            raise ValueError(f"sem_recomendacao deve ser um de {SEM_RECOMENDACAO}: {sem_recomendacao!r}")
        self.modelo = modelo
        self.regra = regra
        self.sem_recomendacao = sem_recomendacao

    def iniciar(self, veiculos, rng):
        """Chamado pelo Simulador na construção, com o rng_motoristas: repassa ao modelo
        de aceite, se ele tiver características por motorista a sortear."""
        if hasattr(self.modelo, "iniciar"):
            self.modelo.iniciar(veiculos, rng)

    def fim_do_passo(self, relatorio):
        """Chamado pelo Simulador no fim de cada passo, depois do relatório: repassa ao
        modelo, se ele evolui com o que aconteceu (ex.: ConfiancaEvolutiva)."""
        if hasattr(self.modelo, "fim_do_passo"):
            self.modelo.fim_do_passo(relatorio)

    def confianca_media(self):
        """A confiança média da frota, se o modelo tiver uma; None caso contrário."""
        return self.modelo.confianca_media() if hasattr(self.modelo, "confianca_media") else None

    @property
    def precisa_preferencias(self):
        return (self.regra.precisa_preferencias or self.sem_recomendacao == "preferencias"
                or getattr(self.modelo, "precisa_preferencias", False))

    @property
    def precisa_previsao(self):
        return getattr(self.modelo, "precisa_previsao", False)

    @property
    def precisa_renda(self):
        return getattr(self.modelo, "precisa_renda", False)

    def responder(self, veiculo, recomendacao, ctx, u_aceite, u_destino):
        """(aceitou, destino). recomendacao=None é "sem recomendação" (aceitou = None).
        Aceita se u_aceite < prob; o u_destino escolhe o destino quando preciso."""
        if recomendacao is None:
            if self.sem_recomendacao == "ficar":
                return None, veiculo.zona
            return None, pelas_preferencias(veiculo, ctx, u_destino)
        if u_aceite < self.modelo.prob(veiculo, recomendacao, ctx):
            return True, recomendacao
        return False, self.regra.destino(veiculo, recomendacao, ctx, u_destino)


# --- o modelo de Ashkrof (1.10c) ---

class ComportamentoAshkrof(Comportamento):
    """Logit multinomial de Ashkrof (Tabela 3; arquivo de decisões, seção 8): o motorista
    escolhe entre W (esperar onde está), H (ir para a zona recomendada) e C (sair rodando
    para uma vizinha). Decide tudo de uma vez, então não usa modelo de aceite nem regra
    pós-recusa: reescreve o responder.

      V_W = 0,207 − 0,022 · min(espera, 25) + 0,322 · 1 (local central) + 0,286 · 0
            (estacionamento) + 0,350 · fim_de_semana
      V_H = 0,080 · min(corridas, 14) − 0,025 · minutos até a zona recomendada
      V_C = −0,312 · familiar (zona atual = base ou vizinha da base)

    espera = (passo atual − fim da última corrida) × 10 min; minutos = tempos.minutos(zona,
    recomendada, hora), 0 na recomendação de ficar (pendência A); "início do turno" não
    entra. P = exp(V) ÷ soma dos exp(V).

    Com os dois números de sempre: u_aceite escolhe a alternativa pela acumulada, na ordem
    W, H, C. Mover: H aceita e vai; W recusa e fica; C recusa e sai rodando. Ficar: W ou H
    aceita e fica; C recusa e sai rodando. Sair rodando: vizinha sorteada com u_destino
    pelas preferências, sem a zona atual e sem a recomendada (renormalizadas); se nenhuma
    sobrar, fica. Sem recomendação: o sem_recomendacao de sempre (o Ashkrof não entra)."""
    precisa_tempos = True
    ALTERNATIVAS = ("W", "H", "C")
    TETO_ESPERA_MIN = 25
    TETO_CORRIDAS = 14

    def __init__(self, sem_recomendacao="ficar"):
        super().__init__(modelo=None, regra=None, sem_recomendacao=sem_recomendacao)

    @property
    def precisa_preferencias(self):
        return True  # o C sorteia a vizinha pelas preferências

    def espera_min(self, veiculo, ctx):
        return (ctx.tempo - veiculo.fim_ultima_corrida) * 10

    def utilidades(self, veiculo, recomendada, ctx):
        espera = min(self.espera_min(veiculo, ctx), self.TETO_ESPERA_MIN)
        fim_de_semana = 1 if ctx.tipo_dia == "sabado" else 0
        minutos = 0.0 if recomendada == veiculo.zona else ctx.tempos.minutos(veiculo.zona, recomendada, ctx.hora)
        familiar = 1 if (veiculo.zona == veiculo.base or veiculo.zona in ctx.grade.vizinhas(veiculo.base)) else 0
        v_w = 0.207 - 0.022 * espera + 0.322 * 1 + 0.286 * 0 + 0.350 * fim_de_semana
        v_h = 0.080 * min(veiculo.corridas_feitas, self.TETO_CORRIDAS) - 0.025 * minutos
        v_c = -0.312 * familiar
        return v_w, v_h, v_c

    def probabilidades(self, veiculo, recomendada, ctx):
        """(P(W), P(H), P(C))."""
        exps = [math.exp(v) for v in self.utilidades(veiculo, recomendada, ctx)]
        total = sum(exps)
        return tuple(e / total for e in exps)

    def sair_rodando(self, veiculo, recomendada, ctx, u):
        """Vizinha pelas preferências, sem a zona atual e sem a recomendada; se não
        sobrar nenhuma (ou todas com preferência zero), fica."""
        pref = ctx.preferencias.de(veiculo.base, veiculo.zona, ctx.hora)
        restantes = [z for z in ctx.grade.vizinhas(veiculo.zona) if z != recomendada]
        total = sum(pref[z] for z in restantes)
        if not restantes or total <= 0:
            return veiculo.zona
        return pela_acumulada(restantes, [pref[z] / total for z in restantes], u)

    def responder(self, veiculo, recomendacao, ctx, u_aceite, u_destino):
        if recomendacao is None:
            return super().responder(veiculo, recomendacao, ctx, u_aceite, u_destino)
        alternativa = pela_acumulada(self.ALTERNATIVAS, self.probabilidades(veiculo, recomendacao, ctx), u_aceite)
        if alternativa == "C":
            return False, self.sair_rodando(veiculo, recomendacao, ctx, u_destino)
        if recomendacao == veiculo.zona:   # ficar: W ou H obedecem
            return True, veiculo.zona
        if alternativa == "H":
            return True, recomendacao
        return False, veiculo.zona         # W: recusa e fica
