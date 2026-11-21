import csv

from environment.grafo_zonas import GrafoZonas
from environment.tempo import espera_min, passos_da_corrida

# Estados do veículo:
#   OCIOSO --matching--> EM_CORRIDA --chegada (n passos)--> OCIOSO
#   OCIOSO --recomendação de mover--> REPOSICIONANDO --passo seguinte--> OCIOSO
OCIOSO = "ocioso"                  # parado numa zona: disponível para matching e recomendações
EM_CORRIDA = "em_corrida"          # levando um passageiro: a duração registrada, em passos
REPOSICIONANDO = "reposicionando"  # indo vazio para uma vizinha: sempre 1 passo

# Vizinhança das grades: cima, baixo, esquerda, direita e diagonais.
# A ordem importa: é a ordem de vizinhas(), e o Hotspot desempata por ela.
DIRECOES = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]


class GradeCelulas(GrafoZonas):
    """Grade com só as células que existem (ex.: as células de Manhattan com viagens).

    celulas: (linha, coluna) de cada célula. Os ids das zonas são 0, 1, 2, ...
        na ordem (linha, coluna) crescente, independente da ordem recebida.
        Num quadrado n x n completo, a célula (i, j) tem id i * n + j.
    nao_vizinhos: pares de células adjacentes que NÃO são vizinhas (ex.: os
        pares com mais de 15 min, como as travessias do Central Park). A
        vizinhança é simétrica: o par (a, b) tira a->b e b->a.
    """

    def __init__(self, celulas, nao_vizinhos=()):
        celulas = [tuple(c) for c in celulas]
        if len(set(celulas)) != len(celulas):
            raise ValueError("células repetidas")
        self._celulas = sorted(celulas)
        self._ids = {c: i for i, c in enumerate(self._celulas)}

        bloqueados = set()
        for a, b in nao_vizinhos:
            a, b = tuple(a), tuple(b)
            if a not in self._ids or b not in self._ids:
                raise ValueError(f"par {a}-{b}: célula que não existe na grade")
            if (b[0] - a[0], b[1] - a[1]) not in DIRECOES:
                raise ValueError(f"par {a}-{b}: as células não são adjacentes")
            bloqueados |= {(a, b), (b, a)}

        self._vizinhas = [
            [self._ids[(l + dl, c + dc)] for dl, dc in DIRECOES
             if (l + dl, c + dc) in self._ids and ((l, c), (l + dl, c + dc)) not in bloqueados]
            for l, c in self._celulas
        ]

    def zonas(self):
        """Ids de todas as zonas: 0, 1, ..., n-1."""
        return list(range(len(self._celulas)))

    def vizinhas(self, zona):
        """Ids das zonas vizinhas, na ordem de DIRECOES."""
        return list(self._vizinhas[zona])

    def celula(self, zona):
        """(linha, coluna) da zona."""
        return self._celulas[zona]

    def zona(self, celula):
        """Id da zona que corresponde à célula (linha, coluna)."""
        return self._ids[tuple(celula)]


class GradeQuadrada(GradeCelulas):
    """Grade quadrada size x size completa: caso particular da GradeCelulas."""

    def __init__(self, size):
        self.size = size  # ex.: size=5  ->  cidade 5x5 = 25 zonas
        super().__init__([(i, j) for i in range(size) for j in range(size)])


def grade_do_csv(caminho, nao_vizinhos=()):
    """GradeCelulas a partir de um CSV com as colunas linha e coluna (ex.: dataset/grade_1200.csv).

    Usa o csv da biblioteca padrão, e não o pandas, para o simulador não depender
    das bibliotecas do ETL. É uma função, e não um método da GradeCelulas, porque
    a GradeQuadrada herdaria o método, e o construtor dela só recebe size.
    """
    with open(caminho, newline="") as f:
        celulas = [(int(r["linha"]), int(r["coluna"])) for r in csv.DictReader(f)]
    return GradeCelulas(celulas, nao_vizinhos)


def nao_vizinhos_do_csv(caminho):
    """Pares de células ((linha, coluna), (linha, coluna)) de um CSV de pares excluídos
    (ex.: dataset/nao_vizinhos_1200.csv, com as colunas linha_a, coluna_a, linha_b, coluna_b)."""
    with open(caminho, newline="") as f:
        return [((int(r["linha_a"]), int(r["coluna_a"])), (int(r["linha_b"]), int(r["coluna_b"])))
                for r in csv.DictReader(f)]


class Veiculo:
    """Um veículo numa zona, em um dos três estados: OCIOSO, EM_CORRIDA ou REPOSICIONANDO.

    destino e chegada só valem durante o reposicionamento: a zona para onde vai e o
    passo em que chega. Até lá, zona continua sendo a de origem.

    base: a zona-base do motorista, fixa durante toda a simulação; é também a zona
    inicial."""

    def __init__(self, id_veiculo, zona):
        self.id = id_veiculo
        self.base = zona
        self.zona = zona
        self.estado = OCIOSO
        self.destino = None
        self.chegada = None
        # estado do motorista para o modelo de Ashkrof (lado do ambiente: a observação
        # dos algoritmos não mostra): corridas com passageiro já feitas e o passo em que
        # a última terminou (a espera não zera num reposicionamento)
        self.corridas_feitas = 0
        self.fim_ultima_corrida = 0

    def __repr__(self):
        return (f"Veiculo(id={self.id}, "
                f"Zona = {self.zona}, "
                f"Estado={self.estado})")

class Pedido:
    """Um pedido que tem uma origem, destino e custo.

    Os pedidos das viagens reais (DemandaRegistros) trazem também o horário real
    (segundo_do_dia), a duração registrada (duracao_min) e a distância (km), usados no
    tempo de espera, na duração das corridas (tarefa 1.5) e na tarifa (1.12). A
    DemandaPoisson não os usa: ficam None."""
    def __init__(self, id_pedido, cliente, origem, destino, custo, timestep,
                 segundo_do_dia=None, duracao_min=None, km=None):
        self.id = id_pedido
        self.cliente = cliente
        self.origem = origem
        self.destino = destino
        self.custo = custo
        self.criado_em = timestep
        self.expira = self.criado_em + 2
        self.segundo_do_dia = segundo_do_dia
        self.duracao_min = duracao_min
        self.km = km
    
    def __repr__(self):
        return (f"Pedido(id={self.id}, "
                f"{self.origem}->{self.destino}, "
                f"custo={self.custo})")
    
class Corrida:
    """Um corrida que tem um veículo e um pedido a ser atendido"""
    def __init__(self, veiculo, pedido, timestep):
        self.veiculo = veiculo
        self.pedido = pedido
        # a duração registrada, em passos (pedidos da Poisson, sem duração: 1 passo)
        self.horario_chegada = timestep + passos_da_corrida(pedido.duracao_min)
        # espera em minutos, do horário real do pedido até o fim do passo do atendimento
        # (timestep); os pedidos da Poisson não têm horário: None
        self.espera_min = (None if pedido.segundo_do_dia is None
                           else espera_min(pedido.segundo_do_dia, timestep - pedido.criado_em))

    def __repr__(self):
        return (
            f"Corrida("
            f"veiculo={self.veiculo.id}, "
            f"pedido={self.pedido.id}, "
            f"origem={self.pedido.origem}, "
            f"destino={self.pedido.destino}, "
            f"chegada={self.horario_chegada})"
        )