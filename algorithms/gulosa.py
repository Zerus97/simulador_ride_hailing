from algorithms.algoritmo_reposicionamento import AlgoritmoReposicionamento


class GulosaDemanda(AlgoritmoReposicionamento):
    """Baseline "gulosa por demanda esperada" (Chen; Brar): cada ocioso vai, entre a zona
    atual e as vizinhas, para a de maior λ no passo SEGUINTE (obs.passo_do_dia + 1),
    porque é quando o carro chega à zona nova. O passo 144 vira o 0 (a previsão trata).

    Desempate pela ordem das candidatas: a zona atual primeiro, depois a ordem de
    grade.vizinhas; num empate, o carro fica.

    Todos os ociosos de uma zona vão para o mesmo lugar, de propósito: esse amontoamento
    é a fraqueza conhecida da gulosa."""

    def decidir(self, obs):
        if obs.previsao is None:
            raise ValueError("a gulosa por demanda esperada precisa da previsão λ (Simulador(previsao=...))")
        proximo = obs.passo_do_dia + 1
        recomendacoes = {}
        for vid, zona in obs.ociosos:
            candidatas = [zona] + list(obs.grade.vizinhas(zona))
            # max devolve a primeira candidata de maior λ: o empate fica com a zona atual
            recomendacoes[vid] = max(candidatas, key=lambda z: obs.previsao.lam(z, proximo))
        return recomendacoes
