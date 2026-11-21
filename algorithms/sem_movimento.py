from algorithms.algoritmo_reposicionamento import AlgoritmoReposicionamento

class SemMovimento(AlgoritmoReposicionamento):
    def decidir(self, obs):
        return {}