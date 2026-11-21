from algorithms.algoritmo_reposicionamento import AlgoritmoReposicionamento


class VizinhoAleatorio(AlgoritmoReposicionamento):
    """Baseline "vizinho aleatório" (Lin; Chen; Brar): cada ocioso é mandado para uma
    vizinha sorteada por igual, sem a zona atual.

    A ordem dos sorteios é parte do contrato: para cada ocioso, na ordem de obs.ociosos
    (ordem de id), um self.rng.integers(len(vizinhas)), com as vizinhas na ordem de
    grade.vizinhas. O gerador é o do algoritmo (iniciar), que não mexe nos pedidos."""

    def decidir(self, obs):
        recomendacoes = {}
        for vid, zona in obs.ociosos:
            vizinhas = list(obs.grade.vizinhas(zona))
            if vizinhas:  # num grafo conexo, toda zona tem vizinha
                recomendacoes[vid] = vizinhas[self.rng.integers(len(vizinhas))]
        return recomendacoes
