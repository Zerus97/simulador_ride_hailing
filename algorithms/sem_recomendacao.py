from algorithms.algoritmo_reposicionamento import AlgoritmoReposicionamento


class SemRecomendacao(AlgoritmoReposicionamento):
    """Baseline "sem recomendação" (AAVR; i-Rebalance; Q8): a plataforma não recomenda
    nada, e cada ocioso se move pelas próprias preferências, sorteadas a cada passo
    (comportamento com sem_recomendacao="preferencias")."""
    exige_sem_recomendacao = "preferencias"

    def decidir(self, obs):
        return {}
