"""Comportamento dos motoristas diante da recomendação (o A da proposta, tarefa 1.9).

Fica no environment/: o comportamento é do ambiente, não do algoritmo. Três peças:
  - ModeloAceite: a probabilidade de aceitar a recomendação (sem sortear);
  - RegraPosRecusa: para onde vai quem recusou;
  - Comportamento: junta as duas e responde um veículo por vez, a partir de dois
    números uniformes que o simulador sorteia SEMPRE (u_aceite e u_destino), usados ou
    não. Assim a quantidade de sorteios não depende do que o modelo decide, e o aceite
    fica encaixado entre valores de ρ (quem aceita com 0,3 também aceita com 0,7).
"""
from dataclasses import dataclass

import numpy as np

SEM_RECOMENDACAO = ("ficar", "preferencias")


@dataclass(frozen=True)
class Contexto:
    """O que o ambiente sabe do motorista na hora da resposta."""
    hora: int              # passo_do_dia // 6 (o resto por 24 fica com Preferencias.de)
    preferencias: object   # environment.preferencias.Preferencias, ou None
    grade: object          # o grafo de zonas


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
        return self.regra.precisa_preferencias or self.sem_recomendacao == "preferencias"

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
