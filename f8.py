from mbo import MBO
import time
import math

def fitness(position):
    time.sleep(0.001)
    sum = 0
    for x_i in position:
        abs_x = abs(x_i)
            
        sqrt_x = math.sqrt(abs_x)

        sin_x = math.sin(sqrt_x)

        term = -x_i*sin_x

        sum += term
    return sum

if __name__ == '__main__':
    start_time = time.time()
    num_dim = 30
    min_bound = -500
    max_bound = 500

    # Numero de soluciones
    NP = 50
    # Numero de iteraciones
    iter = 1000  
    # Ratio de mariposas en zona 1
    p = 5/12
    # Indicador de ajuste 
    # (a mayor bar, mayor probabilidad de que se realice un vuelo de levy)
    bar = 5/12
    # Periodo de migracion 
    # (a mayor peri, mayor probabilidad hay de que se seleccione una mariposa de la zona 2)
    peri = 1.2
    # Salto maximo que una mariposa puede hacer
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