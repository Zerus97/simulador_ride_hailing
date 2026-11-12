class GrafoZonas:
    #Lista de ids, sempre na mesma ordem (a demanda sorteia nessa ordem)
    def zonas(self):
        raise NotImplementedError

    #Ids das vizinhas, sempre na mesma ordem (o Hotspot desempata por ela)
    def vizinhas(self, zona):
        raise NotImplementedError

    #Tempo de viagem em passos; tau(z, z) é 0
    def tau(self, origem, destino):
        raise NotImplementedError