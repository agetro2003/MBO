
import random
import math
import copy

import multiprocessing as mp


'''
 NP is the total number of
the population

P is the ratio of monarch butterflies in Land
1

'''
beta=1.5
    # PASO 1: Calcular Sigma (La parte compleja con función Gamma)
    # Numerador de la fórmula de Mantegna
numerador = math.gamma(1 + beta) * math.sin(math.pi * beta / 2)
    
    # Denominador de la fórmula
denominador = math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)
    
    # Sigma final elevado a la potencia 1/beta
sigma = (numerador / denominador) ** (1 / beta)

def levy_flight():

    # PASO 2: Generar u y v (Distribuciones Normales)
    # u sigue una distribución Normal(0, sigma^2)
    u = random.gauss(0, sigma) 
    
    # v sigue una distribución Normal(0, 1)
    v = random.gauss(0, 1)

    # PASO 3: Calcular el paso final
    step = u / abs(v) ** (1 / beta)
    
    return step


def worker_migration(data):
    index, old_land1, old_land2, num_dim, peri, p, fitness = data

    if index == 0: 
        return None
    
    child_pos = [0.0] * num_dim

    for k in range(num_dim):
        r = random.random() * peri
        if r <= p:
            random_butterfly1 = random.choice(old_land1)
            child_pos[k] = random_butterfly1.position[k]
        else: 
            random_butterfly2 = random.choice(old_land2)
            child_pos[k] = random_butterfly2.position[k]

    child_fitness = fitness(child_pos)

    if child_fitness < old_land1[index].getFitness():
        return (index, child_pos, child_fitness)
    return None
    

def worker_adjusting(data):
    index, old_land1, old_land2, alpha, num_dim, p, bar, fitness = data
    best_butterfly = old_land1[0]
   
    parent = old_land2[index]
    child_pos = [0.0] * num_dim

    dx_vector = [levy_flight() for _ in range(num_dim)]
    for k in range(num_dim):
        rand_val = random.random()
        if rand_val <= p:
            child_pos[k] = best_butterfly.position[k]
        else: 
            random_butterfly = random.choice(old_land2)
            child_pos[k] = random_butterfly.position[k]

            if rand_val > bar:
                
                child_pos[k] = child_pos[k] + alpha * (dx_vector[k] - 0.5)

    child_fitness = fitness(child_pos)

    if child_fitness < parent.getFitness():
        return (index, child_pos, child_fitness)
    return None

    

        
    




class Monarch_Butterfly:
    def __init__(self, num_dim, min_p, max_p):
        self.position = [random.uniform(min_p, max_p) for _ in range(num_dim)]
        self.fitness = None
    
    def setPosition(self, position): 
        self.position = position
    
    def getPosition(self):
        return self.position

    def setFitness(self, fitness):
        self.fitness = fitness
    
    def getFitness(self):
        return self.fitness


class MBO:
    def __init__(self, NP, p, num_dim, min_p, max_p, fitness, bar, peri, iteration, S_max):
        self.NP = NP
        self.land1 = []
        self.land2 = []
        self.p = p
        self.num_dim = num_dim
        self.min_p = min_p
        self.max_p = max_p
        self.fitness = fitness
        self.bar = bar
        self.peri = peri
        self.S_max = S_max
        self.iteration = iteration
        self.land1_length = math.ceil(NP * p)
        self.t = 0

        self.butterflies = [Monarch_Butterfly(num_dim, min_p, max_p) for _ in range(NP)]

        #set fitness for each butterfly
        for butterfly in self.butterflies:
            butterfly.setFitness(fitness(butterfly.getPosition()))

        
    def migration_operator(self, pool):

        old_land1 = copy.deepcopy(self.land1)
        old_land2 = copy.deepcopy(self.land2)

        tasks = []

        for i in range(len(self.land1)):
            
            task_data = (
                i,
                old_land1,
                old_land2,
                self.num_dim,
                self.peri,
                self.p,
                self.fitness
            )

            tasks.append(task_data)

        
        results = pool.map(worker_migration, tasks)

        for res in results:
            if res is not None:
                index, new_pos, new_fitness = res
                self.land1[index].setPosition(new_pos)
                self.land1[index].setFitness(new_fitness)




    def adjusting_operator(self, pool):
        
        old_land1 = copy.deepcopy(self.land1)
        old_land2 = copy.deepcopy(self.land2)

        tasks = []

        #alpha = self.S_max / (self.t ** 2)
        alpha = self.S_max * (1 - (self.t / self.iteration))



        for i in range(len(self.land2)):
            task_data = (
                i,
                old_land1,
                old_land2,
                alpha,
                self.num_dim,
                self.p,
                self.bar,
                self.fitness
            )

            tasks.append(task_data)

        results = pool.map(worker_adjusting, tasks)
        for res in results:
            if res is not None:
                index, new_pos, new_fitness = res
                self.land2[index].setPosition(new_pos)
                self.land2[index].setFitness(new_fitness)


    def cycle(self, pool):
        self.t += 1
        self.butterflies.sort(key=lambda butterfly: butterfly.getFitness())

        self.land1 = self.butterflies[:self.land1_length]
        self.land2 = self.butterflies[self.land1_length:]

        self.migration_operator(pool)
        self.adjusting_operator(pool)

        self.butterflies = self.land1 + self.land2
    
    def run(self): 
        cpu_count = mp.cpu_count()
        print(f"Iniciando MBO Paralelo con {cpu_count} núcleos...")
        
        with mp.Pool(processes=cpu_count) as pool:
            for i in range(self.iteration):
                self.cycle(pool)

        return self.butterflies[0]










