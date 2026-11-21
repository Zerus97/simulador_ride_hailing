class GrafoZonas:
    def zonas(self):
        """Lista de ids, sempre na mesma ordem (a demanda sorteia nessa ordem)"""
        raise NotImplementedError

    def vizinhas(self, zona):
        """Ids das vizinhas, sempre na mesma ordem (o Hotspot desempata por ela)"""
        raise NotImplementedError

    def conexo(self):
        """True se de qualquer zona se chega a qualquer outra andando por vizinhas.

        Busca em profundidade a partir da primeira zona. Usa só zonas() e
        vizinhas(), então vale para qualquer grafo. Um grafo sem zonas conta
        como conexo.
        """
        zonas = self.zonas()
        if not zonas:
            return True
        vistas = {zonas[0]}
        pilha = [zonas[0]]
        while pilha:
            zona = pilha.pop()
            for vizinha in self.vizinhas(zona):
                if vizinha not in vistas:
                    vistas.add(vizinha)
                    pilha.append(vizinha)
        return len(vistas) == len(zonas)
