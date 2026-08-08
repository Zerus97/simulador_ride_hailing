class Grade: # Mudar para hexagonos
    """A cidade dividida em zonas, numa grade quadrada size x size."""

    def __init__(self, size):
        self.size = size  # ex.: size=5  ->  cidade 5x5 = 25 zonas

    def zonas(self):
        """Lista de todas as zonas, cada uma identificada por (linha, coluna)."""
        return [(i, j) for i in range(self.size) for j in range(self.size)]

    def vizinhas(self, zona):
        """Zonas adjacentes (cima, baixo, esquerda, direita) dentro da grade."""
        i, j = zona
        resultado = []
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < self.size and 0 <= nj < self.size:
                resultado.append((ni, nj))
        return resultado


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
    proximo_id = 1
    """Um pedido que tem uma origem, destino e custo"""
    def __init__(self, cliente, origem, destino, custo, timestep):
        self.id = Pedido.proximo_id
        Pedido.proximo_id += 1
        self.cliente = cliente
        self.origem = origem
        self.destino = destino
        self.custo = custo
        self.expira = timestep + 2
    
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