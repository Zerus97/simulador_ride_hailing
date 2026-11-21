from algorithms.contrato import Observacao


class AlgoritmoReposicionamento:  # Classe de algoritmos de reposicionamento
    sequencial = False
    """True: recomenda um motorista por vez e vê a resposta antes do próximo
    (i-Rebalance original, tarefa 4.6). Protocolo previsto (a confirmar na 4.6):
        proxima(obs) -> (id, zona) ou None   # próximo motorista e recomendação; None encerra
        resposta(id, aceitou, zona_final)    # o ambiente sorteia a resposta e informa na hora
    Entre uma chamada e outra, o simulador atualiza a observação.
    Por enquanto o simulador só tem o modo em lote (decidir) e recusa algoritmos
    sequenciais na construção."""

    exige_sem_recomendacao = None
    """O que o motorista sem recomendação precisa fazer para o algoritmo ser o que diz
    ser ("ficar" ou "preferencias"; None = sem exigência). O Simulador confere contra o
    comportamento.sem_recomendacao na construção e levanta ValueError se não bater."""

    def iniciar(self, rng):
        """Chamado pelo Simulador na construção: o gerador próprio do algoritmo (o 5º
        filho do spawn), para sorteios como o vizinho aleatório (1.11) e a exploração dos
        métodos de RL. Sortear com ele não muda os pedidos nem as posições iniciais."""
        self.rng = rng

    def decidir(self, obs: Observacao):
        """Recebe a Observacao do passo e devolve {id do veículo: zona}.

        Só os veículos de obs.ociosos podem receber recomendação, e a zona precisa ser
        a atual (ficar) ou uma vizinha (mover). Omitir o veículo é não recomendar nada;
        None não é permitido. O simulador valida e levanta ValueError."""
        raise NotImplementedError
