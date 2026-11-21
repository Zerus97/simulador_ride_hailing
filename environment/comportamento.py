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
