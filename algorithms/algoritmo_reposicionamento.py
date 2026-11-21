from algorithms.contrato import Observacao


class AlgoritmoReposicionamento:  # Classe de algoritmos de reposicionamento
    def decidir(self, obs: Observacao):
        """Recebe a Observacao do passo e devolve {id do veículo: zona de destino}.
        Só os veículos de obs.ociosos podem receber recomendação."""
        raise NotImplementedError
