"""1.13(a): leitura, validação e montagem de cenários (experimentos/cenario.py)."""
from pathlib import Path

import pandas as pd
import pytest

from environment.previsao import previsao_do_csv
from environment.tempos import tempos_do_csv
from experimentos.cenario import ErroCenario, ler_cenario, montar, tipo_do_dia

DATASET = Path(__file__).resolve().parent.parent / "dataset"

BASE = {
    "cenario": {"nome": "base_util", "finalidade": "avaliacao", "lado": 1200, "dia": 23, "fracao": 0.15,
                "fator_demanda": 1.0, "frota": 2500, "passo_inicial": 18, "passos": 126, "aquecimento": 6},
    "comportamento": {"modelo": "fixo", "rho": 1.0, "regra": "ficar", "sem_recomendacao": "ficar"},
    "algoritmo": {"nome": "sem_movimento"},
    "execucao": {"sementes": [0, 1, 2], "saida": "resultados/base_util"},
}


def toml(dados):
    linhas = []
    for secao, campos in dados.items():
        linhas.append(f"[{secao}]")
        for k, v in campos.items():
            valor = f'"{v}"' if isinstance(v, str) else str(v).lower() if isinstance(v, bool) else repr(v)
            linhas.append(f"{k} = {valor}")
    return "\n".join(linhas) + "\n"


def escrever(tmp_path, mudar=None, tirar=None):
    """O cenário BASE, com campos trocados ({(seção, campo): valor}) ou tirados ([(seção, campo)])."""
    dados = {s: dict(c) for s, c in BASE.items()}
    for (secao, campo), valor in (mudar or {}).items():
        dados[secao][campo] = valor
    for secao, campo in tirar or []:
        del dados[secao][campo]
    caminho = tmp_path / "cenario.toml"
    caminho.write_text(toml(dados))
    return caminho


def test_cenario_valido(tmp_path):
    c = ler_cenario(escrever(tmp_path))
    assert (c.nome, c.dia, c.tipo_dia, c.frota, c.sementes) == ("base_util", 23, "util", 2500, (0, 1, 2))
    assert c.fracao_efetiva == pytest.approx(0.15)


@pytest.mark.parametrize("mudar, tirar, trecho", [
    (None, [("cenario", "frota")], "frota"),                                  # campo faltando
    ({("cenario", "frta"): 10}, None, "frta"),                                # campo desconhecido
    ({("algoritmo", "nome"): "aavr_magico"}, None, "nome"),                    # nome desconhecido
    ({("comportamento", "regra"): "sair_rodando"}, None, "regra"),
    ({("cenario", "lado"): 900}, None, "lado"),                               # faixa
    ({("cenario", "fracao"): "0.15"}, None, "fracao"),                        # tipo
    ({("cenario", "aquecimento"): 200}, None, "aquecimento"),
    (None, [("comportamento", "rho")], "rho"),                                # rho exigido com "fixo"
    ({("comportamento", "modelo"): "questionario"}, None, "rho"),             # rho proibido sem "fixo"
    ({("algoritmo", "nome"): "sem_recomendacao"}, None, "sem_recomendacao"),  # exigência do algoritmo
    ({("comportamento", "n0"): 48.0}, None, "n0"),                             # n0 proibido sem "evolutivo"
    ({("comportamento", "modelo"): "evolutivo"}, [("comportamento", "rho")], "n0"),  # n0 exigido
    ({("comportamento", "modelo"): "ashkrof"}, [("comportamento", "rho")], "regra"),  # regra proibida
    ({("comportamento", "modelo"): "ashkrof"}, [("comportamento", "regra")], "rho"),  # rho proibido
    ({("comportamento", "modelo"): "ashkrof", ("comportamento", "n0"): 48.0},
     [("comportamento", "rho"), ("comportamento", "regra")], "n0"),                    # n0 proibido
], ids=["faltando", "desconhecido", "algoritmo", "regra", "lado", "tipo", "aquecimento",
        "rho_faltando", "rho_sobrando", "exigencia", "n0_sobrando", "n0_faltando",
        "ashkrof_regra", "ashkrof_rho", "ashkrof_n0"])
def test_erros_de_cenario(tmp_path, mudar, tirar, trecho):
    with pytest.raises(ErroCenario, match=trecho):
        ler_cenario(escrever(tmp_path, mudar, tirar))


@pytest.mark.parametrize("dia, tipo", [(23, "util"), (19, "sabado"), (14, "util"), (5, "sabado")])
def test_tipo_do_dia(dia, tipo):
    assert tipo_do_dia(dia) == tipo


@pytest.mark.parametrize("dia", [18, 20, 17, 26, 27])  # sexta, domingo, excluídos, Páscoa
def test_tipo_do_dia_fora_do_desenho(dia):
    with pytest.raises(ErroCenario):
        tipo_do_dia(dia)


@pytest.mark.parametrize("finalidade, dia", [("avaliacao", 8), ("avaliacao", 12), ("avaliacao", 15),
                                             ("ajuste", 23), ("ajuste", 19), ("ajuste", 22), ("ajuste", 31)])
def test_guarda_de_vazamento(tmp_path, finalidade, dia):
    with pytest.raises(ErroCenario):
        ler_cenario(escrever(tmp_path, {("cenario", "finalidade"): finalidade, ("cenario", "dia"): dia}))


def test_avaliacao_em_dia_extra_e_permitida(tmp_path):
    assert ler_cenario(escrever(tmp_path, {("cenario", "dia"): 22})).dia == 22


def test_ajuste_em_dia_de_treino_e_permitido(tmp_path):
    assert ler_cenario(escrever(tmp_path, {("cenario", "finalidade"): "ajuste", ("cenario", "dia"): 8})).dia == 8


# --- coerência da montagem (com os dados) ---

def test_montagem_coerente_no_dia_23(tmp_path):
    arquivos = [DATASET / n for n in ("pedidos_1200.parquet", "grade_1200.csv", "lambda_1200.csv",
                                      "tempos_1200.csv", "fatores_hora.csv", "nao_vizinhos_1200.csv")]
    if not all(a.exists() for a in arquivos):
        pytest.skip("dataset fora do repositório")
    cenario = ler_cenario(escrever(tmp_path, {("cenario", "fator_demanda"): 1.5,
                                              ("comportamento", "modelo"): "questionario",
                                              ("comportamento", "regra"): "proporcional",
                                              ("algoritmo", "nome"): "gulosa"},
                                   tirar=[("comportamento", "rho")]))
    sim = montar(cenario, pd.read_parquet(arquivos[0]), semente=0)
    g = sim.grade

    # demanda e previsão com o MESMO produto fração × fator (0,15 × 1,5 = 0,225)
    assert sim.fonte_demanda.fracao == pytest.approx(0.225) and sim.fonte_demanda.dia == 23
    total = sum(sim.previsao.lam(z, p) for z in g.zonas() for p in range(144))
    assert total / 0.225 == pytest.approx(324_954, abs=1)  # a previsão de dia útil da 1.4b

    # previsão e tempos de dia útil, e não de sábado
    util = previsao_do_csv(arquivos[2], g, "util", 0.15, 1.5)
    sabado = previsao_do_csv(arquivos[2], g, "sabado", 0.15, 1.5)
    amostra = [(z, p) for z in g.zonas() for p in (30, 70, 113)]
    assert all(sim.previsao.lam(z, p) == util.lam(z, p) for z, p in amostra)
    assert any(sim.previsao.lam(z, p) != sabado.lam(z, p) for z, p in amostra)
    t_util = tempos_do_csv(arquivos[3], arquivos[4], g, 1200, "util")
    a, b = sim.tempos.pares()[0]
    assert all(sim.tempos.minutos(a, b, h) == t_util.minutos(a, b, h) for h in range(24))
    assert sim.cenario is cenario and sim.coletor.passos_aquecimento == 6


def test_modelo_evolutivo_no_cenario(tmp_path):
    c = ler_cenario(escrever(tmp_path, {("comportamento", "modelo"): "evolutivo", ("comportamento", "n0"): 48.0},
                             tirar=[("comportamento", "rho")]))
    assert (c.modelo_aceite, c.n0, c.rho) == ("evolutivo", 48.0, None)


def test_modelo_ashkrof_no_cenario(tmp_path):
    c = ler_cenario(escrever(tmp_path, {("comportamento", "modelo"): "ashkrof"},
                             tirar=[("comportamento", "rho"), ("comportamento", "regra")]))
    assert (c.modelo_aceite, c.regra, c.rho, c.n0) == ("ashkrof", None, None, None)


def test_modelo_logistica_no_cenario(tmp_path):
    base = {("comportamento", "modelo"): "logistica", ("comportamento", "coeficientes"): "reajuste",
            ("comportamento", "deterministico"): True}
    c = ler_cenario(escrever(tmp_path, base, tirar=[("comportamento", "rho")]))
    assert (c.modelo_aceite, c.coeficientes, c.deterministico, c.regra) == ("logistica", "reajuste", True, "ficar")
    padrao = ler_cenario(escrever(tmp_path, {("comportamento", "modelo"): "logistica"}, tirar=[("comportamento", "rho")]))
    assert padrao.coeficientes is None  # o montar usa "publicados", o padrão
    with pytest.raises(ErroCenario, match="coeficientes"):
        ler_cenario(escrever(tmp_path, {**base, ("comportamento", "coeficientes"): "xyz"}, tirar=[("comportamento", "rho")]))
    with pytest.raises(ErroCenario, match="deterministico"):
        ler_cenario(escrever(tmp_path, {("comportamento", "deterministico"): True}))
    with pytest.raises(ErroCenario, match="deterministico"):
        ler_cenario(escrever(tmp_path, {**base, ("comportamento", "deterministico"): 1}, tirar=[("comportamento", "rho")]))
