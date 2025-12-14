import random

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
        for _ in range(self.iter):  
            for particle in self.swarm:
                self.getVelocity(particle)
                self.updatePosition(particle)
                current_fit = self.fitness(particle.position)
                if current_fit < particle.past_best_fitness:    
                    particle.past_best_fitness = current_fit
                    particle.past_best_position = list(particle.position)
                    if current_fit < self.best_global_value:
                        self.best_global_value = current_fit
                        self.best_global_position = list(particle.position)
        return self.best_global_position, self.best_global_value
        



