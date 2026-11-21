from collections import Counter, deque
import numpy as np
from algorithms.contrato import Observacao, RelatorioDoPasso
from environment.env import EM_CORRIDA, OCIOSO, REPOSICIONANDO, Corrida, Pedido, Veiculo

class Simulador:
    """Junta a grade e os veiculos, e faz o tempo avancar em passos."""

    def __init__(self, grade, num_veiculos, demanda, algoritmo, seed=None, previsao=None, tempos=None):
        """demanda: a fonte de demanda (ex.: DemandaPoisson), que diz quais pedidos
        chegam a cada passo e define o calendário (passo_do_dia); ver simulator/demanda.py.
        previsao e tempos: só repassados aos algoritmos, na observação
        (environment/previsao.py e environment/tempos.py)."""
        # O modo sequencial (um motorista por vez, i-Rebalance original) só existe na
        # tarefa 4.6: melhor falhar na construção do que no meio da simulação.
        if getattr(algoritmo, "sequencial", False):
            raise NotImplementedError("modo sequencial: tarefa 4.6")
        # Um gerador por fonte de acaso. Os filhos 0, 1 e 2 de spawn(4) são os mesmos
        # de spawn(3): acrescentar o do matching não muda posições, demanda nem aceite.
        sementes = np.random.SeedSequence(seed).spawn(4)
        self.rng_inicial = np.random.default_rng(sementes[0])
        self.rng_demanda = np.random.default_rng(sementes[1])
        self.rng_aceite  = np.random.default_rng(sementes[2])
        self.rng_matching = np.random.default_rng(sementes[3])
        self.grade = grade
        self.previsao = previsao
        self.tempos = tempos
        self.tempo = 0
        self.fonte_demanda = demanda
        self.pedidos = {zona: deque() for zona in self.grade.zonas()}
        self.historico_pedidos = []
        self.corridas_ativas = []
        self.corridas_concluidas = []
        self.algoritmo = algoritmo
        self.proximo_id_pedido = 1
        # relatório do passo: o que aconteceu, registrado durante o passo
        self.ultimo_relatorio = None   # o do último passo completo (o do passo final nunca chega ao algoritmo)
        self._casamentos, self._chegadas, self._aceites = [], [], []
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
        """Pergunta à fonte de demanda quais pedidos chegam neste passo e cria os Pedidos.
        O gerador entregue a ela é o rng_demanda, o mesmo de antes da fonte existir."""
        for item in self.fonte_demanda.gerar(self.tempo, self.grade, self.rng_demanda):
            novo_pedido = Pedido(self.proximo_id_pedido, None, item["origem"], item["destino"],
                                 item.get("custo"), self.tempo,
                                 segundo_do_dia=item.get("segundo_do_dia"),
                                 duracao_min=item.get("duracao_min"), km=item.get("km"))
            self.pedidos[item["origem"]].append(novo_pedido)
            self.historico_pedidos.append(novo_pedido)
            self.total_pedidos += 1
            self.proximo_id_pedido += 1


    def matching(self):
        """Liga carros ociosos a pedidos da mesma zona, com o motorista sorteado.

        Sortear evita que os carros de id baixo fiquem sempre com a corrida, o que
        distorceria a história de cada motorista (corridas feitas, aceite, ganhos).

        A ordem dos sorteios é parte do contrato (a reprodutibilidade depende dela):
        para cada zona, na ordem de grade.zonas(), com k = min(pedidos na fila, carros
        ociosos) > 0, um único sorteio rng_matching.choice(len(livres), size=k,
        replace=False), com os ociosos da zona em ordem de id. O 1º sorteado leva o
        pedido mais antigo, o 2º o seguinte, e assim por diante."""
        livres_por_zona = {}
        for v in self.veiculos:  # ordem de id
            if v.estado == OCIOSO:
                livres_por_zona.setdefault(v.zona, []).append(v)

        for zona in self.grade.zonas():
            fila = self.pedidos[zona]
            livres = livres_por_zona.get(zona, [])
            k = min(len(fila), len(livres))
            if k == 0:
                continue
            for i in self.rng_matching.choice(len(livres), size=k, replace=False):
                v = livres[i]
                v.estado = EM_CORRIDA
                pedido = fila.popleft()
                self.corridas_ativas.append(Corrida(v, pedido, self.tempo))
                self._casamentos.append((v.id, pedido.id, zona))
    
    def arrivals(self):
        """Libera as corridas que chegam neste passo e os carros que terminam de se
        reposicionar: os dois ficam OCIOSOS na zona de destino, antes do matching."""
        restantes = []
        for corrida in self.corridas_ativas:
            if self.tempo == corrida.horario_chegada:
                corrida.veiculo.zona = corrida.pedido.destino
                corrida.veiculo.estado = OCIOSO
                self.corridas_concluidas.append(corrida)
                self._chegadas.append((corrida.veiculo.id, corrida.veiculo.zona))
            else:
                restantes.append(corrida)
        self.corridas_ativas = restantes

        for v in self.veiculos:
            if v.estado == REPOSICIONANDO and v.chegada == self.tempo:
                v.zona, v.estado, v.destino, v.chegada = v.destino, OCIOSO, None, None
                self._chegadas.append((v.id, v.zona))

    def repositioning(self, moves):
        """Aplica a saída do algoritmo, {id do veículo: zona}:
          - zona vizinha: mover; o carro sai REPOSICIONANDO e chega no passo seguinte;
          - zona atual: ficar (é uma recomendação: na 1.9, o motorista pode recusar);
          - id ausente: sem recomendação (na 1.9, conduta própria do motorista);
          - None: não permitido.
        Tudo é validado antes de aplicar qualquer movimento (ValueError, e não assert:
        o Python ignora asserts com -O). Ainda não há recusa (tarefa 1.9): toda
        recomendação, de mover ou de ficar, é aceita e entra em aceites."""
        self.validar_recomendacoes(moves)
        for veiculo in self.veiculos:  # ordem de id
            if veiculo.id not in moves:
                continue
            destino = moves[veiculo.id]
            self._aceites.append(veiculo.id)
            if destino != veiculo.zona:
                veiculo.estado = REPOSICIONANDO
                veiculo.destino = destino
                veiculo.chegada = self.tempo + 1

    def validar_recomendacoes(self, moves):
        """Cada (id, destino) precisa ser de um veículo ocioso, para a zona atual dele
        ou uma vizinha. Vale para qualquer algoritmo, não só para os baselines."""
        ociosos = {v.id: v.zona for v in self.veiculos if v.estado == OCIOSO}
        for vid, destino in moves.items():
            if vid not in ociosos:
                existe = any(v.id == vid for v in self.veiculos)
                motivo = "não está ocioso" if existe else "não existe"
                raise ValueError(f"recomendação para o veículo {vid!r}, que {motivo}")
            if destino is None:
                raise ValueError(f"recomendação None para o veículo {vid}: para ficar, devolva a "
                                 f"zona atual; para não recomendar, omita o veículo")
            zona = ociosos[vid]
            if destino != zona and destino not in self.grade.vizinhas(zona):
                raise ValueError(f"recomendação inválida para o veículo {vid}: {zona} -> {destino!r} "
                                 f"não é a zona atual nem uma vizinha")

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
        self._casamentos, self._chegadas, self._aceites = [], [], []
        self.demanda()
        self.arrivals()
        self.expire()
        self.matching()
        repositioning = self.algoritmo.decidir(self.observar())  # leva o relatório de t - 1
        self.repositioning(repositioning)
        self.ociosidade()
        self.ultimo_relatorio = RelatorioDoPasso(
            tempo=self.tempo,
            casamentos=tuple(self._casamentos),
            chegadas=tuple(self._chegadas),
            aceites=tuple(self._aceites),
            recusas=(),  # ainda não há recusa (tarefa 1.9)
        )

    def ociosidade(self):
        """Soma os carros sem passageiro: os OCIOSOS e os REPOSICIONANDO, como antes dos
        estados (um carro movido continuava ocioso no fim do passo). Separar parado de
        reposicionando é decisão das métricas (tarefa 1.12)."""
        self.ociosidades = (self.ociosidades + sum(1 for v in self.veiculos if v.estado != EM_CORRIDA))


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

    def observar(self):
        """A Observacao que o algoritmo recebe neste passo (depois do matching).
        Tudo é montado de novo a cada passo: o algoritmo não alcança o estado do simulador."""
        chegadas = Counter()
        for corrida in self.corridas_ativas:
            chegadas[(corrida.pedido.destino, corrida.horario_chegada)] += 1
        return Observacao(
            tempo=self.tempo,
            ociosos=tuple(self.listar_veiculos(ociosos=True)),
            pedidos_abertos=self.contar_pedidos_por_zona(),
            chegadas=chegadas,
            grade=self.grade,
            relatorio=self.ultimo_relatorio,
            passo_do_dia=self.fonte_demanda.passo_do_dia(self.tempo),
            previsao=self.previsao,
            tempos=self.tempos,
        )

    def listar_veiculos(self, ociosos=True):
        return [
            (v.id, v.zona)
            for v in self.veiculos
            if (v.estado == OCIOSO) == ociosos
        ]