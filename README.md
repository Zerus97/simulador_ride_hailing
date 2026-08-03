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

### 2\. Testar o funcionanamento do simulador

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
```python
class MeuAlgoritmo(AlgoritmoReposicionamento):
   def decidir(self, veiculos_ociosos, corridas_ativas, pedidos, grade):
        d = dict()
        return d
```
Este método recebe como input:
- veiculos_ociosos: [(id_veiculo, (zona)), (6, (7,4))]
- corridas_ativas: ({(zona_destino) : numero de carros, (6,1): 4})
- pedidos: {(zona): numero de pedidos, (0,1): 2}
- grade: Uma instancia da grade do simulador

O retorno deste método deve ser um dicionário contendo cada veículo a ser reposicionado(seu id) e a zona para qual zona deve ir no formado (i, j).