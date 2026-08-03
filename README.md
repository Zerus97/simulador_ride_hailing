# Simulador de Ride-Hailing para Algoritmos de reposicionamento
Este programa foi pensado para avalair os diferentes tipos de algoritmos de reposicionamento dentro de um ambiente unificado possibilitando a comparação entre as diferentes soluções existentes.

Requerimentos:
- numpy
- Python 3

## Instalação
 
Clone o repositório, entre no diretório do projeto e instale as dependências:

```bash
git clone https://github.com/Zerus97/simulador_ride_hailing.git
cd simulador_ride_hailing
pip install numpy
```

-----

## Passo a Passo para Execução

### 1\. Executar o comando acima dentro do terminal para clonar o projeto

### 2\. Testar o funcionanamento do simulador

Para testar o funcionamento do simulador execute:
```bash
python3 simulador.py
```

A mensagem de simulador criado seguido das métricas de avaliação devem aparecer.

### 3\. Escolhendo o baseline

Dentro do simulador, existem dois baselines de algoritmos de reposicionamento disponíveis:
| Baseline | Comportamento |
|----------|---------------|
| **SemMovimento** | Os carros permanecem parados enquanto estão ociosos. |
| **HotSpot** | Os carros se movem em direção à zona vizinha com o maior número de pedidos. |

### 4\. Criando seu próprio algoritmo

Dentro do simulador existe uma classe chamada `AlgoritmoReposicionamento` com um método chamado `decidir()`.
Para implementar seu algoritmo basta criar uma nova classe que herde ela e implementar o método a seguir:
```python
class MeuAlgoritmo(AlgoritmoReposicionamento):
   def decidir(self, veiculos_ociosos, corridas_ativas, pedidos, grade):
        d = dict()
        #sua lógica aqui
        return d
```
Este método recebe como input:
| Parâmetro | Descrição | Exemplo |
|-----------|-----------|---------|
| `veiculos_ociosos` | Lista de tuplas `(id_veiculo, zona)` com os veículos disponíveis, onde `zona` é uma tupla `(i, j)`. | `[(6, (7, 4))]` |
| `corridas_ativas` | Dicionário no formato `{zona_destino: numero_de_carros}`. | `{(6, 1): 4}` |
| `pedidos` | Dicionário no formato `{zona: numero_de_pedidos}`. | `{(0, 1): 2}` |
| `grade` | Instância da grade do simulador. | — |

O retorno deste método deve ser um dicionário onde cada chave representa o id do veiculo a ser reposicionado(seu id) e a zona para qual zona deve ir no formado (i, j).