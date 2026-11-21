from algorithms.contrato import Observacao


class AlgoritmoReposicionamento:  # Classe de algoritmos de reposicionamento
    def decidir(self, obs: Observacao):
        """Recebe a Observacao do passo e devolve {id do veículo: zona}.

        Só os veículos de obs.ociosos podem receber recomendação, e a zona precisa ser
        a atual (ficar) ou uma vizinha (mover). Omitir o veículo é não recomendar nada;
        None não é permitido. O simulador valida e levanta ValueError."""
        raise NotImplementedError
