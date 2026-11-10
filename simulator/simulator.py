from collections import Counter
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
        self.pedidos = [[[] for _ in range(self.grade.size)] for _ in range(self.grade.size)]
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
        num_pedidos = 0
        zonas = self.grade.zonas()
        for i in range(self.grade.size):
            for j in range(self.grade.size):
                num_pedidos = int(self.rng_demanda.poisson(self.lam))
                self.total_pedidos += num_pedidos
                for k in range(num_pedidos):
                    self.pedidos[i][j].append(Pedido(self.proximo_id_pedido, None, (i, j), self.escolher([z for z in zonas if z != (i, j)], self.rng_demanda), k + 1, self.tempo))
                    self.proximo_id_pedido += 1
        

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

    def repositioning(self, moves):
        for veiculo in self.veiculos:
            destino = moves.get(veiculo.id)
            if destino is not None:
                veiculo.zona = destino

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
            # self.mostrar()

    def mostrar(self):
        print(f"t={self.tempo}")
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
        if self.total_pedidos == 0:
            taxa_atendimento = float("nan")
        else:
            taxa_atendimento = (len(self.corridas_ativas) + len(self.corridas_concluidas)) / (self.total_pedidos)
        corridas_completadas = len(self.corridas_concluidas)
        if len(self.veiculos) == 0:
            taxa_ociosidade = float("nan")
        else:
            taxa_ociosidade = self.ociosidades / (self.tempo * len(self.veiculos))
        return {
            "taxa_atendimento": taxa_atendimento,
            "pedidos_totais": self.total_pedidos,
            "pedidos_expirados": self.pedidos_expirados,
            "taxa_pedidos_expirados": self.pedidos_expirados / self.total_pedidos,
            "corridas_completadas": corridas_completadas,
            "taxa_ociosidade": taxa_ociosidade,
        }

    def contar_pedidos_por_zona(self):
            return {
                (i, j): len(self.pedidos[i][j])
                for i in range(self.grade.size)
                for j in range(self.grade.size)
            }

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