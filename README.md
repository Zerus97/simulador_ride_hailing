# Simulador de Ride-Hailing para Algoritmos de reposicionamento
Este programa foi pensado para avalair os diferentes tipos de algoritmos de reposicionamento dentro de um ambiente unificado possibilitando a comparação entre as diferentes soluções existentes.

Requerimentos:
- numpy

## Instalação
Para instalar basta rodar o comando:
1. `git clone https://github.com/Zerus97/simulador_ride_hailing.git`

-----

## Passo a Passo para Execução

### 1\. Executar o comando acima dentro do terminal para clonar o projeto

### 2\. Testar o funcionando do simulador

Para testar o funcionamento do simulador execute:
`python3 simulador.py`

A mensagem de simulador criado seguido das métricas de avaliação devem aparecer.

### 3\. Escolhendo o baseline

Dentro do simulador, existem dois baselines de algoritmos de reposicionamento disponíveis:
- SemMovimento(Carros não se mexem enquanto ociosos)
- HotSpot(Carros se movem em direção a zona vizinha com mais pedidos)

### 4\. Criando seu próprio algoritmo

Dentro do simulador existe uma classe chamada AlgoritmoReposicionamento com um método chamado decidir().
Para implementar seu algoritmo basta criar uma nova classe referenciando ela e implementar o método a seguir:
`class MeuAlgoritmo(AlgoritmoReposicionamento):`
`   def decidir(self, veiculos_ociosos, corridas_ativas, pedidos, grade):`
`        d = dict()`
`        return d`

O retorno deste método deve ser um dicionário contendo cada veículo a ser reposicionado(seu id) e a zona para qual deve ir no formado (i, j).