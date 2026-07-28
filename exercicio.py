import numpy as np
import scipy as sp

# Mostra a quantidade de carros em cada loja no estilo (n0, n1) com n1 e n2 indo de 0 até 20.
S = np.indices((21, 21)).transpose(1, 2, 0)

# Ações positivas demonstram carros indo de 0 para 1 e ações negativas demonstram carros indo de 1 para 0. Ações neutras representam carros que não se movem.
A = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]

def transicao(estado, acao):
    carros_0 = estado[0]
    carros_1 = estado[1]

    if acao == 0: # Não move os carros
        novo_estado = [carros_0, carros_1]
        custo = 0

    elif acao > 0: # Move os carros da Loja 0 para a Loja 1
        novo_estado_0 = carros_0 - acao
        novo_estado_1 = carros_1 + acao
        novo_estado = [novo_estado_0, novo_estado_1]
        custo = 2 * acao

    else: # Move os carros da Loja 1 para a Loja 0
        acao = abs(acao) 
        novo_estado_0 = carros_0 + acao
        novo_estado_1 = carros_1 - acao
        novo_estado =  [novo_estado_0, novo_estado_1]
        custo = 2 * acao

    return [novo_estado, custo]

def acoes_validas(estado):
    carros_0 = estado[0] 
    carros_1 = estado[1]

    vagas_0 = 20 - carros_0 
    vagas_1 = 20 - carros_1 

    acao_0 = min(carros_0, vagas_1, 5) # Verifica o limite de ação possível para a Loja 0 entre os carros disponíveis para mandar, o número de vagas disponíveis e considera
    acao_1 = min(carros_1, vagas_0, 5) # o limite de 5

    return list(range(-acao_1, acao_0 + 1))

# A distribuição de Poisson é usada para modelar quantas vezes um evento acontece em um intervalo de tempo, quando:
#- os eventos acontecem de forma aleatória;
#- eles são independentes entre si;
#- existe uma média conhecida de eventos por intervalo.
def poisson(n, lam): # Qual a probabilidade de acontecerem exatamente n alugueis, sabendo que a média é lam alugueis por dia?
    resultado = ((np.exp(-lam)) * (lam **n)) / (sp.special.factorial(n))
    return resultado

def carros_alugados(d, c):
    return min(d, c)

def receita_esperada(c, lam):
    receita = 0
    for d in range(21):
        receita = receita + (poisson(d, lam) * 10 * carros_alugados(d, c))
    return receita

def dist_apos_alugueis(c, lam): # Dado a quantidade de carros que você tem e a média de carros alugados por dia, calcula a probabilidade sobrar k(0..20) carros.
    dpa = np.zeros(21)
    for k in range(c + 1):
        dpa[k] = poisson(c - k, lam)
    # for k in range(c + 1, 21): # Funciona para este caso onde lam = 3 e 4 mas para números grande o erro vai acumular.
    #     dpa[0] = dpa[0] + poisson(k, lam)
    dpa[0] += 1 - dpa.sum() # Forma mais robusta de lidar com a cauda
    return dpa

def dist_final(c, lam_pedidos, lam_retornos):
    sobras = dist_apos_alugueis(c, lam_pedidos)
    final = np.zeros(21)
    for s in range(21):
        for r in range(21):
            retorno = poisson(r, lam_retornos)
            final[min(s + r, 20)] = final[min(s + r, 20)] + sobras[s] * retorno
    return final

def q_valor(estado, acao, V):
    gamma = 0.9
    # 1. aplica a movimentação (transicao) -> estado pós-mov + custo
    novo_estado, custo = transicao(estado, acao)

    # 2. receita esperada das duas lojas - custo  = recompensa esperada
    receita = receita_esperada(novo_estado[0], 3) + receita_esperada(novo_estado[1], 4) - custo

    # 3. dist_final de cada loja
    loja_0 = dist_final(novo_estado[0], 3, 3)
    loja_1 = dist_final(novo_estado[1], 4, 2)

    # 4. loop duplo sobre (f1, f2) acumulando d1[f1]*d2[f2]*V[f1][f2]
    valor_futuro = 0
    for i in range(21):
        for j in range(21):
            valor_futuro = valor_futuro + loja_0[i] * loja_1[j] * V[i, j]
    
    # 5. retorna recompensa + gamma * (essa soma)
    return receita + gamma * valor_futuro

def avaliar_politica(V, politica):
    theta = 1e-4
    while True:
        delta = 0
        for i in range(21):
            for j in range(21):
                v = V[i, j]
                V[i, j] = q_valor([i, j], politica[i, j], V)
                delta = max(delta, abs(v - V[i, j]))
        if delta < theta:
            break
    return V

def melhorar_politica(V, politica):
    policy_stable = True
    for i in range(21):
        for j in range(21):
            acao_antiga = politica[i, j]
            acoes = acoes_validas([i, j])
            melhor_acao = None
            melhor_valor = float('-inf')
            for a in acoes:
                q = q_valor([i, j], a, V)
                if q > melhor_valor:
                    melhor_valor = q
                    melhor_acao = a
            if melhor_acao != acao_antiga:
                policy_stable = False
            politica[i, j] = melhor_acao
    return politica, policy_stable

# print(transicao([2, 19], 1))

# print(acoes_validas([10, 0]))

# print(poisson(4, 4))   # deve dar ~0.195 (moda da Poisson(4))
# print(sum(poisson(n, 4) for n in range(20)))  # deve dar ~1.0

# print(receita_esperada(20, 4))   # esperado ~40

# print(f"{dist_apos_alugueis(5, 3).sum():.10f}")   # loja quase sempre atende a demanda
# print(f"{dist_apos_alugueis(2, 4).sum():.10f}")    # loja pequena, demanda alta

print(dist_final(10, 3, 3))
d = dist_final(10, 3, 3)
print("carro mais provável no fim do dia:", d.argmax())
print("massa em torno de 9-11:", d[9:12].sum())
print(f"{dist_final(10, 3, 3).sum():.10f}")   # tem que dar ~1.0)
for carros, prob in enumerate(d):
    print(f"{carros:2d} carros -> {prob:.4f}")

V = np.zeros((21, 21))
print(V)
print(q_valor([10, 10], 0, V))   # com V toda zero, deve dar só a receita esperada (~gamma*0)

# Avalia a política
V = np.zeros((21, 21))
politica = np.zeros((21, 21), dtype=int)
V = avaliar_politica(V, politica)
print(V[10, 10], V[0, 0], V[20, 20])
print(V)

# Melhora a política
politica, policy_stable = melhorar_politica(V, politica)

#Policy Iteration
V = np.zeros((21, 21))
politica = np.zeros((21, 21), dtype=int)
while True:
    V = avaliar_politica(V, politica)          # Fase 1: avalia a política atual
    politica, estavel = melhorar_politica(V, politica)  # Fase 2: melhora
    if estavel:                                 # ninguém mudou de ação → ótimo
        break