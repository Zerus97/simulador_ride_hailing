# simulador.py — versao minima (incremento 1)
# Objetivo: montar o esqueleto que roda "vazio":
# uma grade de zonas, alguns veiculos, e um laco de tempo.

import random
import numpy as np

class Grade:
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

class Pedido:
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
    
class Corrida:
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

class Simulador:
    """Junta a grade e os veiculos, e faz o tempo avancar em passos."""

    def __init__(self, grade, num_veiculos, lam):
        self.grade = grade
        self.tempo = 0
        self.lam = lam
        self.pedidos = [[[] for _ in range(self.grade.size)] for _ in range(self.grade.size)]
        self.corridas_ativas = []
        self.corridas_concluidas = []
        # espalha os veiculos em zonas aleatorias da grade
        zonas = grade.zonas()
        self.ociosidades = 0
        self.total_pedidos = 0
        self.pedidos_expirados = 0
        self.veiculos = [
            Veiculo(v, random.choice(zonas)) for v in range(num_veiculos)
        ]

    def demanda(self):
        num_pedidos = 0
        zonas = self.grade.zonas()
        for i in range(self.grade.size):
            for j in range(self.grade.size):
                num_pedidos = np.random.poisson(self.lam)
                self.total_pedidos += num_pedidos
                for k in range(num_pedidos):
                    self.pedidos[i][j].append(Pedido(None, (i, j), random.choice([z for z in zonas if z != (i, j)]), k + 1, self.tempo))
        

    def matching(self):
        for v in self.veiculos:
            if v.ocioso:
                i, j = v.zona
                if self.pedidos[i][j]:
                    v.ocioso = False
                    self.corridas_ativas.append(Corrida(v, self.pedidos[i][j][0], self.tempo))
                    self.pedidos[i][j].pop(0)
    
    def arrivals(self):
        restantes = []
        for corrida in self.corridas_ativas:
            if self.tempo == corrida.horario_chegada:
                corrida.veiculo.zona = corrida.pedido.destino
                corrida.veiculo.ocioso = True
                self.corridas_concluidas.append(corrida)
            else:
                restantes.append(corrida)
        self.corridas_ativas = restantes

    def expire(self):
        for i in range(self.grade.size):
            for j in range(self.grade.size):
                novas_lista = []
                for pedido in self.pedidos[i][j]:
                    if pedido.expira > self.tempo:
                        novas_lista.append(pedido)
                    else:
                        self.pedidos_expirados += 1
                self.pedidos[i][j] = novas_lista

    def passo(self):
        """Um passo de tempo. Por enquanto so avanca o relogio e reporta."""
        self.tempo += 1
        self.demanda()
        self.arrivals()
        self.expire()
        self.matching()
        self.ociosidade()
        n_ociosos = sum(1 for v in self.veiculos if v.ocioso)

        # print(f"t={self.tempo}: {n_ociosos} veiculos ociosos de {len(self.veiculos)}")

    def ociosidade(self):
        """Calcula a ociosidade"""
        self.ociosidades = (self.ociosidades + sum(1 for v in self.veiculos if v.ocioso))


    def rodar(self, passos):
        for _ in range(passos):
            self.passo()
            # self.mostrar()

    def mostrar(self):
        for i in range(self.grade.size):
            for j in range(self.grade.size):
                zona = (i, j)
                veiculos_na_zona = [v.id for v in self.veiculos if v.zona == zona]
                pedidos_na_zona = len(self.pedidos[i][j])
                corridas_na_zona = [c for c in self.corridas_ativas if c.pedido.origem == zona]
                print(f"Zona {zona}: Veiculos {veiculos_na_zona} Pedidos {pedidos_na_zona} Corridas {corridas_na_zona}")

    def mostrar_v2(self):
        ocupacao = {}
        for v in self.veiculos:
            ocupacao.setdefault(v.zona, []).append(v.id)

        for i in range(self.grade.size):
            linha = []
            for j in range(self.grade.size):
                ids = ocupacao.get((i, j), [])
                celula = ",".join(str(id_) for id_ in ids) if ids else "."
                linha.append(f"{celula:>4}")
            print(" ".join(linha))

    def metrics(self):
        pedidos_atendidos = (len(self.corridas_ativas) + len(self.corridas_concluidas)) / (self.total_pedidos)
        corridas_completadas = len(self.corridas_concluidas)
        taxa_ociosidade = self.ociosidades / (self.tempo * len(self.veiculos))
        print(f"Pedidos Atendidos: {pedidos_atendidos}, "
              f"Pedidos Totais {self.total_pedidos}, "
              f"Pedidos Expirados {self.pedidos_expirados}, "
              f"Taxa de Pedidos Expirados {self.pedidos_expirados / self.total_pedidos}, " 
              f"Corridas Completadas: {corridas_completadas}, "
              f"Taxa de Ociosidade: {taxa_ociosidade}")



if __name__ == "__main__":
    random.seed(0)  # fixa a aleatoriedade para o resultado ser sempre igual
    grade = Grade(size=5)                 # cidade 5x5 = 25 zonas
    sim = Simulador(grade, num_veiculos=10, lam=0.40)
    print(f"Simulador criado: {len(grade.zonas())} zonas, {len(sim.veiculos)} veiculos")
    sim.rodar(passos=500)
    sim.metrics()
    # sim.mostrar()
    # sim.mostrar_v2()