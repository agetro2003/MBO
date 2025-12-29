from parallel_MBO import MBO
import time
import math
def fitness(position): 
    sum_val = 0
    for i in range(len(position)):
        # Suma todos los elementos desde el inicio hasta i+1 (no inclusivo)
        inner_sum = sum(position[:i+1]) 
        sum_val += inner_sum**2
    return sum_val

if __name__ == '__main__':
    start_time = time.time()
    num_dim = 30
    min_bound = -100
    max_bound = 100

    NP = 50
    iter = 1000  


    p = 5/12

    bar = 5/12

    peri = 1.2

    S_max = 2

    mbo = MBO(NP, p, num_dim, min_bound, max_bound, fitness, bar, peri, iter, S_max)

    best_butterfly = mbo.run()
    best_position = best_butterfly.getPosition()
    best_value = best_butterfly.getFitness()
    end_time = time.time()
    result_time = end_time - start_time
    print(best_position)
    print(best_value)
    print("Tiempo de ejecución: " + f"{result_time:.3f}" + " segundos")