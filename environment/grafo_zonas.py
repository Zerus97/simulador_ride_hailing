class GrafoZonas:
    def zonas(self):
        """Lista de ids, sempre na mesma ordem (a demanda sorteia nessa ordem)"""
        raise NotImplementedError

    def vizinhas(self, zona):
        """Ids das vizinhas, sempre na mesma ordem (o Hotspot desempata por ela)"""
        raise NotImplementedError