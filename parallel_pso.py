import random

import multiprocessing as mp

def getVelocity(position, velocity, w, local_weight, num_dim, past_best_position, global_weight, best_global_position):
    new_velocity = [0.0] * num_dim

    for i in range(num_dim):
        r1 = random.random()
        r2 = random.random()
        inertia = w * velocity[i]
        local = local_weight * r1 * (past_best_position[i] - position[i])
        global_ = global_weight * r2 * (best_global_position[i] - position[i])
        new_velocity[i] = inertia + local + global_


    return new_velocity

def updatePosition(position, velocity, num_dim, min_p, max_p): 
    new_position = [0.0] * num_dim

    for i in range(num_dim):
        new_position[i] = position[i] + velocity[i]
        if new_position[i] < min_p:
            new_position[i] = min_p
        elif new_position[i] > max_p:
            new_position[i] = max_p

        
    return new_position


def worker_pso(data):
    position, velocity, w, local_weight, global_weight, min_p, max_p, index, fitness, past_best_fitness, best_global_position, past_best_position = data

    num_dim = len(position)

    new_velocity = getVelocity(position, velocity, w, local_weight, num_dim, past_best_position, global_weight, best_global_position)

    new_position = updatePosition(position, new_velocity, num_dim, min_p, max_p)

    current_fit = fitness(new_position)

    new_best_position = past_best_position
    new_best_fitness = past_best_fitness

    if current_fit < past_best_fitness:
        new_best_position = list(new_position)
        new_best_fitness = current_fit
    
    return (index, new_position, new_velocity, new_best_position, new_best_fitness)

    
        



class Particle:
    def __init__(self, num_dim, min_p, max_p):
        self.position = [random.uniform(min_p, max_p) for _ in range(num_dim)]
        self.velocity = [0 for _ in range(num_dim) ]
        self.past_best_position = list(self.position)
        self.past_best_fitness = float('inf')


class PSO: 
    def __init__(self, num_dim, min_p, max_p, num_particles, iter, w, local_weight, global_weight, fitness):
        self.num_dim = num_dim
        self.min_p = min_p
        self.max_p = max_p
        self.num_particles = num_particles
        self.iter = iter
        self.w = w
        self.local_weight = local_weight
        self.global_weight = global_weight
        self.fitness = fitness
        
        self.best_global_value = float('inf')
        self.best_global_position = [float('inf') for _ in range(num_dim)]
        self.swarm = []
        for _ in range(num_particles):
            particle = Particle(num_dim, min_p, max_p)
            current_fit = self.fitness(particle.position)
            particle.past_best_fitness = current_fit
            self.swarm.append(particle)
            if self.best_global_value > current_fit:
                self.best_global_value = current_fit
                self.best_global_position = list(particle.position)
    
    def getVelocity(self, particle):
        for i in range(self.num_dim):
            r1 = random.random()
            r2 = random.random()
            inertia = self.w*particle.velocity[i]
            local = self.local_weight*r1*(particle.past_best_position[i]-particle.position[i])
            global_ = self.global_weight*r2*(self.best_global_position[i]-particle.position[i])
            particle.velocity[i] = inertia + local + global_

    def updatePosition(self, particle):
        for i in range(self.num_dim):
            particle.position[i] += particle.velocity[i]
            if particle.position[i] < self.min_p:
                particle.position[i] = self.min_p
            elif particle.position[i] > self.max_p:
                particle.position[i] = self.max_p

    def run(self):

        cpu_count = mp.cpu_count()
        print(f"Iniciando PSO con {cpu_count} procesos")

        with mp.Pool(processes=cpu_count) as pool:
            
            for it in range(self.iter):
                tasks = []

                for i, particle in enumerate(self.swarm):
                    task = (
                        particle.position,
                        particle.velocity,
                        self.w,
                        self.local_weight,
                        self.global_weight,
                        self.min_p,
                        self.max_p,
                        i,
                        self.fitness,
                        particle.past_best_fitness,
                        self.best_global_position,
                        particle.past_best_position
                    )
                    tasks.append(task)
                
                results = pool.map(worker_pso, tasks)

                for res in results:
                    idx, new_position, new_velocity, new_best_position, new_best_fitness = res
                    particle = self.swarm[idx]
                    particle.position = new_position
                    particle.velocity = new_velocity
                    particle.past_best_position = new_best_position
                    particle.past_best_fitness = new_best_fitness
                    if new_best_fitness < self.best_global_value:
                        self.best_global_value = new_best_fitness
                        self.best_global_position = new_best_position



        return self.best_global_position, self.best_global_value
        



