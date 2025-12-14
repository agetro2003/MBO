from parallel_MBO import MBO
import time

def fitness(position): 
    #time.sleep(0.001)
    absolute_values = [abs(x_i) for x_i in position]
    return max(absolute_values)
    
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