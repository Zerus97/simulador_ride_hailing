from collections import Counter, deque
import numpy as np
from environment.env import Veiculo, Pedido, Corrida

class Simulador:
    """Junta a grade e os veiculos, e faz o tempo avancar em passos."""

    def __init__(self, grade, num_veiculos, lam, algoritmo, seed=None): # Como definir o lam de forma que represente melhor a realidade?
        sementes = np.random.SeedSequence(seed).spawn(3)
        self.rng_inicial = np.random.default_rng(sementes[0])
        self.rng_demanda = np.random.default_rng(sementes[1])
        self.rng_aceite  = np.random.default_rng(sementes[2]) 
        self.grade = grade
        self.tempo = 0
        self.lam = lam
        self.pedidos = {zona: deque() for zona in self.grade.zonas()}
        self.historico_pedidos = []
        self.corridas_ativas = []
        self.corridas_concluidas = []
        self.algoritmo = algoritmo
        self.proximo_id_pedido = 1
        # espalha os veiculos em zonas aleatorias da grade
        zonas = grade.zonas()
        self.ociosidades = 0
        self.total_pedidos = 0
        self.pedidos_expirados = 0
        self.veiculos = [
            Veiculo(v, self.escolher(zonas, self.rng_inicial)) for v in range(num_veiculos)
        ]

    def escolher(self, opcoes, rng):
        """Escolhe um elemento aleatorio da lista usando o gerador informado."""
        return opcoes[rng.integers(len(opcoes))]

    def demanda(self):
        zonas = self.grade.zonas()
        for zona in zonas:
            num_pedidos = int(self.rng_demanda.poisson(self.lam))
            self.total_pedidos += num_pedidos
            for k in range(num_pedidos):
                novo_pedido = Pedido(self.proximo_id_pedido, None, zona, self.escolher([z for z in zonas if z != zona], self.rng_demanda), k + 1, self.tempo)
                self.pedidos[zona].append(novo_pedido)
                self.historico_pedidos.append(novo_pedido) # <--- Salva apenas a instância
                self.proximo_id_pedido += 1


    def matching(self):
        for v in self.veiculos:
            if v.ocioso and self.pedidos[v.zona]:
                v.ocioso = False
                pedido = self.pedidos[v.zona].popleft()
                self.corridas_ativas.append(Corrida(v, pedido, self.tempo))
    
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

    def repositioning(self, moves):
        for veiculo in self.veiculos:
            destino = moves.get(veiculo.id)
            if destino is not None:
                veiculo.zona = destino

    def expire(self):
        for zona, lista in self.pedidos.items():
            nova_lista = deque()
            for pedido in lista:
                if pedido.expira > self.tempo:
                    nova_lista.append(pedido)
                else:
                    self.pedidos_expirados += 1
            self.pedidos[zona] = nova_lista

    def passo(self):
        self.tempo += 1
        self.demanda()
        self.arrivals()
        self.expire()
        self.matching()
        repositioning = self.algoritmo.decidir(self.listar_veiculos(ociosos=True),
                               self.contar_veiculos_por_zona_futura(ociosos=False), 
                               self.contar_pedidos_por_zona(),
                               self.grade)
        self.repositioning(repositioning)
        self.ociosidade()

    def ociosidade(self):
        """Calcula a ociosidade"""
        self.ociosidades = (self.ociosidades + sum(1 for v in self.veiculos if v.ocioso))


    def rodar(self, passos):
        for _ in range(passos):
            self.passo()

    def taxa(self, numerator, denominator):
        if denominator == 0:
            return float("nan")
        else:
            return numerator / denominator

    def metrics(self):
        taxa_atendimento = self.taxa(len(self.corridas_ativas) + len(self.corridas_concluidas), (self.total_pedidos))
        corridas_completadas = len(self.corridas_concluidas)
        taxa_ociosidade = self.taxa(self.ociosidades, (self.tempo * len(self.veiculos)))
        taxa_pedidos_expirados = self.taxa(self.pedidos_expirados, self.total_pedidos)
        return {
            "taxa_atendimento": taxa_atendimento,
            "pedidos_totais": self.total_pedidos,
            "pedidos_expirados": self.pedidos_expirados,
            "taxa_pedidos_expirados": taxa_pedidos_expirados,
            "corridas_completadas": corridas_completadas,
            "taxa_ociosidade": taxa_ociosidade,
        }

    def contar_pedidos_por_zona(self):
        return {zona: len(lista) for zona, lista in self.pedidos.items()}

    def pedidos_abertos(self):
        return [pedido for lista in self.pedidos.values() for pedido in lista]

    def contar_veiculos_por_zona_futura(self, ociosos=True):
        contagem = Counter()
        for corrida in self.corridas_ativas:
            contagem[corrida.pedido.destino] += 1
        return contagem

    def listar_veiculos(self, ociosos=True):
        return [
            (v.id, v.zona)
            for v in self.veiculos
            if v.ocioso == ociosos
        ]