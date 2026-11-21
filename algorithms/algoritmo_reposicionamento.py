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

    def decidir(self, obs: Observacao):
        """Recebe a Observacao do passo e devolve {id do veículo: zona}.

        Só os veículos de obs.ociosos podem receber recomendação, e a zona precisa ser
        a atual (ficar) ou uma vizinha (mover). Omitir o veículo é não recomendar nada;
        None não é permitido. O simulador valida e levanta ValueError."""
        raise NotImplementedError
