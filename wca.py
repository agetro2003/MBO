import random
class Solution: 
    def __init__(self, num_dim, min_p, max_p):
        self.position = [random.uniform(min_p, max_p) for _ in range(num_dim)]
        self.fitness = float('inf')
        self.role = None
        self.parent : Solution = None
    
    def setRole(self, role):
        self.role = role
    
    def setParent(self, parent):
        self.parent = parent

    def move_to_parent(self, c=0.2, max = 100, min = 100):
        if self.parent is not None:
            for i in range(len(self.position)):
                self.position[i] += c * random.random() * (self.parent.position[i] - self.position[i])
                if self.position[i] > max:
                    self.position[i] = max
                elif self.position[i] < min:
                    self.position[i] = min





def distribute_streams(streams, rivers):
    # 1. SI NO HAY RÍOS O ARROYOS, SALIR
    if not rivers or not streams:
        return

    # 2. CALCULAR DISTRIBUCIÓN BASADA EN RANGO (Más seguro que 1/fitness)
    # El mejor río (índice 0) obtiene el mayor peso
    num_rivers = len(rivers)
    num_streams = len(streams)
    
    # Creamos pesos basados en ranking: 
    # Si hay 3 ríos, pesos = [3, 2, 1]. El primero se lleva más.
    ranks = list(range(num_rivers, 0, -1)) 
    total_rank = sum(ranks)
    
    distribution = []
    assigned_count = 0
    
    for r in range(num_rivers):
        # Calcular cantidad proporcional
        count = int(round((ranks[r] / total_rank) * num_streams))
        distribution.append(count)
        assigned_count += count

    # 3. CORRECCIÓN DE ERRORES DE REDONDEO
    # Si faltan por asignar, se los damos al mejor río (el primero)
    while assigned_count < num_streams:
        distribution[0] += 1
        assigned_count += 1
    
    # Si sobran (asignamos de más), se los quitamos al último río
    while assigned_count > num_streams:
        for i in range(num_rivers - 1, -1, -1):
            if distribution[i] > 0:
                distribution[i] -= 1
                assigned_count -= 1
                if assigned_count == num_streams:
                    break
    
    # 4. ASIGNACIÓN FINAL (CON PROTECCIÓN DE ÍNDICE)
    stream_index = 0
    for r in range(num_rivers):
        limit = distribution[r]
        for _ in range(limit):
            # --- PROTECCIÓN CRÍTICA ---
            if stream_index >= num_streams:
                break
            # --------------------------
            
            streams[stream_index].setParent(rivers[r])
            stream_index += 1
            
    # Caso borde: si por alguna razón extraña quedó alguno sin padre
    # (por ejemplo, lógica anterior falló), asignar el resto al mejor río.
    while stream_index < num_streams:
        streams[stream_index].setParent(rivers[0])
        stream_index += 1

def distribute_agents(agents, num_of_rivers):
    sorted_agents = sorted(agents, key=lambda x: x.fitness)
    sea = sorted_agents[0]
    sea.setRole("sea")
    rivers = sorted_agents[1:num_of_rivers+1]
    for river in rivers:
        river.setRole("river")
        river.setParent(sea)

    streams = sorted_agents[num_of_rivers+1:]
    for stream in streams:
        stream.setRole("stream")
    distribute_streams(streams, rivers)
        
def distance(agent_a, agent_b):
    return sum((a - b) ** 2 for a, b in zip(agent_a.position, agent_b.position)) ** 0.5



class WCA:
    def __init__(self, num_agents, iter, num_dim, min_p, max_p, fitness, num_of_rivers, min_distance = 0.01, epsilon = 10, w = 0.1):
        self.num_agents = num_agents
        self.num_dim = num_dim
        self.min_p = min_p
        self.max_p = max_p
        self.fitness = fitness
        self.agents: list[Solution] = [] 
        self.min_distance = min_distance
        self.epsilon = epsilon
        self.iter = iter
        self.w = w


        for _ in range(num_agents):
            agent = Solution(num_dim, min_p, max_p)
            agent.fitness = fitness(agent.position)
            self.agents.append(agent)
        distribute_agents(self.agents, num_of_rivers)

    def promote_stream_to_river(self, stream : Solution):
        old_river = stream.parent
        sea = old_river.parent
        
        stream.setRole("river")
        stream.setParent(sea)
      
        old_river.setRole("stream")
        old_river.setParent(stream)

        for agent in self.agents:
            if agent.parent == old_river:
                agent.setParent(stream)

    def promote_river_to_sea(self, river : Solution):
        sea = river.parent

        rivers_of_sea = [a for a in self.agents 
                         if a.parent == sea and a is not river]

        streams_of_river = [a for a in self.agents 
                            if a.parent == river]


        river.setRole("sea")
        river.setParent(None)

        sea.setRole("river")
        sea.setParent(river)

        for r in rivers_of_sea:
            r.setParent(river)
        for s in streams_of_river:
            s.setParent(sea)

    def check_promotions(self):

        for agent in self.agents:
            if agent.role == "stream":
                if agent.fitness < agent.parent.fitness:
                    self.promote_stream_to_river(agent)
        for agent in self.agents:
            if agent.role == "river":
                if agent.fitness < agent.parent.fitness:
                    self.promote_river_to_sea(agent)

    def evaporation_agents(self):
        for agent in self.agents:
            if agent.role == "stream":
                if agent.fitness > agent.parent.fitness * self.epsilon or distance(agent, agent.parent) < self.min_distance:
                    agent.position = [random.uniform(self.min_p, self.max_p) for _ in range(self.num_dim)]
                    agent.fitness = self.fitness(agent.position)
            if agent.role == "river":
                if distance(agent, agent.parent) < self.min_distance:
                    agent.position = [random.uniform(self.min_p, self.max_p) for _ in range(self.num_dim)]
                    agent.fitness = self.fitness(agent.position)



    def cycle(self):
        for agent in self.agents:
            agent.move_to_parent(self.w)
            agent.fitness = self.fitness(agent.position)
        self.check_promotions()
        self.evaporation_agents()


    def get_solution(self):
        for _ in range(self.iter):
            self.cycle()
        return min(self.agents, key=lambda x: x.fitness)
    
    def run(self):
        return self.get_solution()







        
    