from algorithms.hotspot import Hotspot
from algorithms.sem_movimento import SemMovimento
import argparse
import random
from environment.env import Grade
from simulator.simulator import Simulador

if __name__ == "__main__":
    ALGORITMOS = {
    "sem_movimento": SemMovimento,
    "hotspot": Hotspot,
    #"meu_algoritmo" : MeuAlgoritmo
    }
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=5)
    parser.add_argument("--veiculos", type=int, default=10)
    parser.add_argument("--lam", type=float, default=0.4)
    parser.add_argument("--passos", type=int, default=100)
    parser.add_argument(
    "--algoritmo",
    choices=ALGORITMOS.keys(),
    default="sem_movimento",
    help="Algoritmo de reposicionamento"
)

    args = parser.parse_args()
    algoritmo = ALGORITMOS[args.algoritmo]()

    random.seed(0)  # fixa a aleatoriedade para o resultado ser sempre igual
    grade = Grade(size=args.size)                 # Exemplo cidade 5x5 = 25 zonas
    sim = Simulador(grade, num_veiculos=args.veiculos, lam=args.lam, algoritmo=algoritmo)
    print(f"Simulador criado: {len(grade.zonas())} zonas, {len(sim.veiculos)} veiculos")
    sim.rodar(passos=args.passos)
    sim.metrics()