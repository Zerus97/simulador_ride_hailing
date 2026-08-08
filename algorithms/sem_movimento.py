from algorithms.algoritmo_reposicionamento import AlgoritmoReposicionamento

class SemMovimento(AlgoritmoReposicionamento):
    def decidir(self, veiculos_ociosos, corridas_ativas, pedidos, grade):
        d = dict()
        return d