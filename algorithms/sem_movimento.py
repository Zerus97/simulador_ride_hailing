from algorithms.algoritmo_reposicionamento import AlgoritmoReposicionamento


class SemMovimento(AlgoritmoReposicionamento):
    """Baseline "sem reposicionamento" (Pavone; Lin): nenhuma recomendação, e os ociosos
    ficam onde estão."""
    exige_sem_recomendacao = "ficar"

    def decidir(self, obs):
        return {}
