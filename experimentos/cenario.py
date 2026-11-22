"""Cenários de experimento: um arquivo TOML descreve o cenário, e montar() constrói tudo.

É a camada de cima (como o main.py): importa todas as outras e lê o dataset; nada
importa de volta daqui. Montar tudo a partir de um único cenário evita combinar peças
que não combinam (a previsão de sábado com um dia útil; frações diferentes na demanda e
na previsão) e deixa registrado o que foi rodado.

Exemplo de arquivo:

    [cenario]
    nome = "base_util"
    finalidade = "avaliacao"      # ou "ajuste" (busca de β do AAVR nos dias de treino)
    lado = 1200
    dia = 23
    fracao = 0.15
    fator_demanda = 1.0
    frota = 2500
    passo_inicial = 18            # 3h
    passos = 126                  # até 24h
    aquecimento = 6

    [comportamento]
    modelo = "fixo"               # ou "questionario"
    rho = 1.0                     # só com modelo = "fixo"
    regra = "ficar"               # ou "proporcional"
    sem_recomendacao = "ficar"    # ou "preferencias"

    [algoritmo]
    nome = "sem_movimento"

    [execucao]
    sementes = [0, 1, 2]
    saida = "resultados/base_util"
"""
import tomllib
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from algorithms.gulosa import GulosaDemanda
from algorithms.hotspot import Hotspot
from algorithms.sem_movimento import SemMovimento
from algorithms.sem_recomendacao import SemRecomendacao
from algorithms.vizinho_aleatorio import VizinhoAleatorio
from environment.comportamento import (COEFICIENTES_LOGISTICA, AceiteFixo, AceiteRecomendacao, Comportamento,
                                       ComportamentoAshkrof, ConfiancaEvolutiva,
                                       ConfiancaQuestionario,
                                       Ficar, Proporcional,
                                       UniformeQuatroPreferidas, ZonaPreferida)
from environment.env import grade_do_csv, nao_vizinhos_do_csv
from environment.preferencias import DIAS_POR_TIPO, preferencias_do_csv
from environment.previsao import previsao_do_csv
from environment.renda import renda_do_csv
from environment.tempo import PASSOS_POR_DIA
from environment.tempos import tempos_do_csv
from environment.calendario import (DIAS_AVALIACAO, DIAS_AVALIACAO_EXTRAS, DIAS_TREINO, DIAS_VALIDACAO,
                                    SABADOS_TREINO, DiaForaDoDesenho)
from environment.calendario import tipo_do_dia as _tipo_do_dia
from simulator.demanda import DemandaRegistros
from simulator.simulator import Simulador

# a pasta dataset/ a partir da raiz do projeto (sem importar o etl/, que puxa a geografia)
DATASET = Path(__file__).resolve().parent.parent / "dataset"

ALGORITMOS = {
    "sem_movimento": SemMovimento,
    "sem_recomendacao": SemRecomendacao,
    "vizinho_aleatorio": VizinhoAleatorio,
    "gulosa": GulosaDemanda,
    "hotspot": Hotspot,
}
MODELOS_ACEITE = ("fixo", "questionario", "evolutivo", "ashkrof", "logistica")
REGRAS = {"ficar": Ficar, "proporcional": Proporcional, "preferida": ZonaPreferida,
          "quatro_preferidas": UniformeQuatroPreferidas}
LADOS = (1200, 600)
FINALIDADES = ("avaliacao", "ajuste")


class ErroCenario(ValueError):
    """Cenário inválido: campo faltando, desconhecido, de tipo ou faixa errados."""


@dataclass(frozen=True)
class Cenario:
    nome: str
    finalidade: str
    lado: int
    dia: int
    fracao: float
    fator_demanda: float
    frota: int
    passo_inicial: int
    passos: int
    aquecimento: int
    modelo_aceite: str
    rho: float | None
    n0: float | None
    deterministico: bool | None
    coeficientes: str | None
    regra: str | None
    sem_recomendacao: str
    algoritmo: str
    sementes: tuple
    saida: str

    @property
    def tipo_dia(self):
        return tipo_do_dia(self.dia)

    @property
    def fracao_efetiva(self):
        """fração × fator de demanda: o MESMO produto na demanda e na previsão."""
        return self.fracao * self.fator_demanda


def tipo_do_dia(dia):
    """O tipo_do_dia do calendário (environment/calendario.py), com o erro como ErroCenario."""
    try:
        return _tipo_do_dia(dia)
    except DiaForaDoDesenho as erro:
        raise ErroCenario(str(erro)) from None


# --- leitura e validação ---

def _campo(secao, dados, nome, tipo, condicao=None, descricao="", obrigatorio=True):
    if nome not in dados:
        if obrigatorio:
            raise ErroCenario(f"[{secao}] falta o campo {nome!r}")
        return None
    valor = dados[nome]
    tipos = (int, float) if tipo is float else tipo
    if (tipo is not bool and isinstance(valor, bool)) or not isinstance(valor, tipos):
        raise ErroCenario(f"[{secao}] {nome} = {valor!r}: esperado {getattr(tipo, '__name__', tipo)}")
    if condicao is not None and not condicao(valor):
        raise ErroCenario(f"[{secao}] {nome} = {valor!r}: {descricao}")
    return valor


def _sem_extras(secao, dados, conhecidos):
    extras = set(dados) - set(conhecidos)
    if extras:
        raise ErroCenario(f"[{secao}] campo desconhecido: {sorted(extras)}")


def validar(dados):
    """Cenario a partir do dicionário lido do TOML; ErroCenario com mensagem clara."""
    secoes = ("cenario", "comportamento", "algoritmo", "execucao")
    _sem_extras("raiz", dados, secoes)
    for secao in secoes:
        if secao not in dados:
            raise ErroCenario(f"falta a seção [{secao}]")
    c, b, a, e = (dados[s] for s in secoes)

    campos_c = {
        "nome": (str, lambda v: v.strip() != "", "não pode ser vazio"),
        "finalidade": (str, lambda v: v in FINALIDADES, f"um de {FINALIDADES}"),
        "lado": (int, lambda v: v in LADOS, f"um de {LADOS}"),
        "dia": (int, lambda v: 1 <= v <= 31, "de 1 a 31"),
        "fracao": (float, lambda v: 0 < v <= 1, "em (0, 1]"),
        "fator_demanda": (float, lambda v: v > 0, "maior que 0"),
        "frota": (int, lambda v: v > 0, "maior que 0"),
        "passo_inicial": (int, lambda v: 0 <= v < PASSOS_POR_DIA, f"de 0 a {PASSOS_POR_DIA - 1}"),
        "passos": (int, lambda v: v > 0, "maior que 0"),
        "aquecimento": (int, lambda v: v >= 0, "0 ou mais"),
    }
    _sem_extras("cenario", c, campos_c)
    vc = {n: _campo("cenario", c, n, *spec) for n, spec in campos_c.items()}
    if vc["aquecimento"] >= vc["passos"]:
        raise ErroCenario("[cenario] o aquecimento precisa ser menor que o número de passos")

    _sem_extras("comportamento", b, ("modelo", "rho", "n0", "deterministico", "coeficientes", "regra",
                                     "sem_recomendacao"))
    modelo = _campo("comportamento", b, "modelo", str, lambda v: v in MODELOS_ACEITE, f"um de {MODELOS_ACEITE}")
    rho = _campo("comportamento", b, "rho", float, lambda v: 0 <= v <= 1, "em [0, 1]",
                 obrigatorio=(modelo == "fixo"))
    if modelo != "fixo" and rho is not None:
        raise ErroCenario("[comportamento] rho só vale com modelo = \"fixo\"")
    n0 = _campo("comportamento", b, "n0", float, lambda v: v > 0, "maior que 0",
                obrigatorio=(modelo == "evolutivo"))
    if modelo != "evolutivo" and n0 is not None:
        raise ErroCenario("[comportamento] n0 só vale com modelo = \"evolutivo\"")
    deterministico = _campo("comportamento", b, "deterministico", bool, obrigatorio=False)
    coeficientes = _campo("comportamento", b, "coeficientes", str, lambda v: v in COEFICIENTES_LOGISTICA,
                          f"um de {tuple(COEFICIENTES_LOGISTICA)}", obrigatorio=False)
    if modelo != "logistica" and (deterministico is not None or coeficientes is not None):
        raise ErroCenario("[comportamento] deterministico e coeficientes só valem com modelo = \"logistica\"")
    if modelo == "logistica" and (coeficientes or "publicados") not in COEFICIENTES_LOGISTICA:
        raise ErroCenario(f"[comportamento] coeficientes = {coeficientes or 'publicados'!r} ainda não "
                          f"registrados; disponíveis: {tuple(COEFICIENTES_LOGISTICA)}")
    # o Ashkrof decide sozinho para onde vai quem recusa: sem regra pós-recusa
    regra = _campo("comportamento", b, "regra", str, lambda v: v in REGRAS, f"um de {tuple(REGRAS)}",
                   obrigatorio=(modelo != "ashkrof"))
    if modelo == "ashkrof" and regra is not None:
        raise ErroCenario("[comportamento] regra não vale com modelo = \"ashkrof\" (ele decide o destino da recusa)")
    sem = _campo("comportamento", b, "sem_recomendacao", str,
                 lambda v: v in ("ficar", "preferencias"), "\"ficar\" ou \"preferencias\"")

    _sem_extras("algoritmo", a, ("nome",))
    algoritmo = _campo("algoritmo", a, "nome", str, lambda v: v in ALGORITMOS, f"um de {tuple(ALGORITMOS)}")

    exige = ALGORITMOS[algoritmo].exige_sem_recomendacao
    if exige is not None and exige != sem:
        raise ErroCenario(f"[algoritmo] {algoritmo} exige [comportamento] sem_recomendacao = {exige!r}")

    _sem_extras("execucao", e, ("sementes", "saida"))
    sementes = _campo("execucao", e, "sementes", list,
                      lambda v: v and all(isinstance(s, int) and not isinstance(s, bool) for s in v),
                      "lista não vazia de inteiros")
    saida = _campo("execucao", e, "saida", str, lambda v: v.strip() != "", "não pode ser vazio")

    cenario = Cenario(**vc, modelo_aceite=modelo, rho=None if rho is None else float(rho),
                      n0=None if n0 is None else float(n0), deterministico=deterministico,
                      coeficientes=coeficientes, regra=regra,
                      sem_recomendacao=sem, algoritmo=algoritmo, sementes=tuple(sementes), saida=saida)
    cenario.tipo_dia  # sexta, domingo e dias excluídos dão erro já na leitura
    _guarda_de_vazamento(cenario)
    return cenario


def _guarda_de_vazamento(cenario):
    """Ninguém avalia num dia que o modelo já viu, e o ajuste não usa os dias de
    avaliação, nem os principais nem os extras."""
    vistos = set(DIAS_TREINO) | set(SABADOS_TREINO) | set(DIAS_VALIDACAO)
    if cenario.finalidade == "avaliacao" and cenario.dia in vistos:
        raise ErroCenario(f"avaliação no dia {cenario.dia}/mar, que é de treino ou de validação")
    if cenario.finalidade == "ajuste" and cenario.dia in set(DIAS_AVALIACAO) | set(DIAS_AVALIACAO_EXTRAS):
        raise ErroCenario(f"ajuste no dia {cenario.dia}/mar, que é de avaliação")


def ler_cenario(caminho):
    with open(caminho, "rb") as f:
        try:
            dados = tomllib.load(f)
        except tomllib.TOMLDecodeError as erro:
            raise ErroCenario(f"{caminho}: TOML inválido: {erro}") from None
    return validar(dados)


# --- montagem ---

@cache
def _pecas(lado, tipo_dia, fracao, fator_demanda):
    """As peças fixas de um cenário (lidas uma vez por combinação)."""
    grade = grade_do_csv(DATASET / f"grade_{lado}.csv",
                         nao_vizinhos=nao_vizinhos_do_csv(DATASET / f"nao_vizinhos_{lado}.csv"))
    previsao = previsao_do_csv(DATASET / f"lambda_{lado}.csv", grade, tipo_dia, fracao, fator_demanda)
    tempos = tempos_do_csv(DATASET / f"tempos_{lado}.csv", DATASET / "fatores_hora.csv", grade, lado, tipo_dia)
    preferencias = preferencias_do_csv(DATASET / f"lambda_{lado}.csv", grade, lado, tipo_dia,
                                       DIAS_POR_TIPO[tipo_dia])
    renda = renda_do_csv(DATASET / f"renda_{lado}.csv", grade, tipo_dia)
    return grade, previsao, tempos, preferencias, renda


def montar(cenario, pedidos, semente):
    """Simulador pronto para o cenário e a semente. pedidos: o pedidos_{lado}.parquet."""
    grade, previsao, tempos, preferencias, renda = _pecas(cenario.lado, cenario.tipo_dia,
                                                   cenario.fracao, cenario.fator_demanda)
    demanda = DemandaRegistros(pedidos, cenario.dia, cenario.fracao_efetiva, cenario.passo_inicial)
    if cenario.modelo_aceite == "ashkrof":
        comportamento = ComportamentoAshkrof(sem_recomendacao=cenario.sem_recomendacao)
    else:
        modelo = {"fixo": lambda: AceiteFixo(cenario.rho), "questionario": ConfiancaQuestionario,
                  "evolutivo": lambda: ConfiancaEvolutiva(cenario.n0),
                  "logistica": lambda: AceiteRecomendacao(
                      deterministico=bool(cenario.deterministico),
                      coeficientes=cenario.coeficientes or "publicados")}[cenario.modelo_aceite]()
        comportamento = Comportamento(modelo, REGRAS[cenario.regra](), sem_recomendacao=cenario.sem_recomendacao)
    sim = Simulador(grade, cenario.frota, demanda, ALGORITMOS[cenario.algoritmo](), seed=semente,
                    previsao=previsao, tempos=tempos, preferencias=preferencias,
                    comportamento=comportamento, passos_aquecimento=cenario.aquecimento, renda=renda)
    sim.cenario = cenario
    return sim
