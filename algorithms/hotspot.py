from algorithms.algoritmo_reposicionamento import AlgoritmoReposicionamento
    
class Hotspot(AlgoritmoReposicionamento):
    def decidir(self, veiculos_ociosos, corridas_ativas, pedidos, grade):
        d = dict()
        for v in veiculos_ociosos:
            melhor_zona = max(
                (z for z in grade.vizinhas(v[1]) if pedidos[z] > 0),
                key=lambda z: pedidos[z],
                default=None
            )
            d[v[0]] = melhor_zona
        return d