from algorithms.algoritmo_reposicionamento import AlgoritmoReposicionamento
    
class Hotspot(AlgoritmoReposicionamento):
    def decidir(self, obs):
        pedidos = obs.pedidos_abertos
        d = dict()
        for vid, zona in obs.ociosos:
            melhor_zona = max(
                (z for z in obs.grade.vizinhas(zona) if pedidos[z] > 0),
                key=lambda z: pedidos[z],
                default=zona  # nenhuma vizinha com pedido: recomenda ficar
            )
            d[vid] = melhor_zona
        return d