from environment.grafo_zonas import GrafoZonas

# Vizinhança das grades: cima, baixo, esquerda, direita e diagonais.
# A ordem importa: é a ordem de vizinhas(), e o Hotspot desempata por ela.
DIRECOES = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]


class GradeQuadrada(GrafoZonas):
    """A cidade dividida em zonas, numa grade quadrada size x size."""

    def __init__(self, size):
        self.size = size  # ex.: size=5  ->  cidade 5x5 = 25 zonas

    def zonas(self):
        """Lista de todas as zonas, cada uma identificada por um id inteiro."""
        return [(i * self.size + j) for i in range(self.size) for j in range(self.size)]

    def vizinhas(self, zona):
        """Zonas adjacentes (cima, baixo, esquerda, direita e diagonais) dentro da grade."""
        i, j = divmod(zona, self.size)
        resultado = []
        for di, dj in DIRECOES:
            ni, nj = i + di, j + dj
            if 0 <= ni < self.size and 0 <= nj < self.size:
                resultado.append(ni * self.size + nj)
        return resultado


class GradeCelulas(GrafoZonas):
    """Grade com só as células que existem (ex.: as células de Manhattan com viagens).

    celulas: (linha, coluna) de cada célula. Os ids das zonas são 0, 1, 2, ...
        na ordem (linha, coluna) crescente, independente da ordem recebida.
        Com todas as células de um quadrado n x n, os ids e as vizinhas saem
        iguais aos da GradeQuadrada(n).
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


class Veiculo:
    """Um veiculo que fica numa zona e pode estar ocioso ou ocupado."""

    def __init__(self, id_veiculo, zona):
        self.id = id_veiculo
        self.zona = zona
        self.ocioso = True
    
    def __repr__(self):
        return (f"Veiculo(id={self.id}, "
                f"Zona = {self.zona}, "
                f"Ocioso={self.ocioso})")

class Pedido: # Custo precisa ser calculado baseado na distancia entre origem e destino
    """Um pedido que tem uma origem, destino e custo"""
    def __init__(self, id_pedido, cliente, origem, destino, custo, timestep):
        self.id = id_pedido
        self.cliente = cliente
        self.origem = origem
        self.destino = destino
        self.custo = custo
        self.criado_em = timestep
        self.expira = self.criado_em + 2
    
    def __repr__(self):
        return (f"Pedido(id={self.id}, "
                f"{self.origem}->{self.destino}, "
                f"custo={self.custo})")
    
class Corrida: # Como calcular horário de chegada?
    """Um corrida que tem um veículo e um pedido a ser atendido"""
    def __init__(self, veiculo, pedido, timestep):
        self.veiculo = veiculo
        self.pedido = pedido
        self.horario_chegada = timestep + 1

    def __repr__(self):
        return (
            f"Corrida("
            f"veiculo={self.veiculo.id}, "
            f"pedido={self.pedido.id}, "
            f"origem={self.pedido.origem}, "
            f"destino={self.pedido.destino}, "
            f"chegada={self.horario_chegada})"
        )