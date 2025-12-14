import cocoex  # La librería oficial de experimentos de BBOB
import numpy as np
import time

# --- IMPORTACIÓN DE TUS ALGORITMOS ---
# Asegúrate de que los archivos 'parallel_MBO.py' y 'pso_parallel.py' 
# están en la misma carpeta que este script.
from parallel_MBO import MBO
from parallel_pso import PSO

def run_experiment(algorithm_name, budget_multiplier=1000):
    """
    Función principal que ejecuta el benchmark completo.
    
    Args:
        algorithm_name (str): "MBO" o "PSO".
        budget_multiplier (int): Cuántas evaluaciones permitimos por dimensión.
                                 Ej: Si dim=10 y mult=1000 -> 10,000 evaluaciones.
    """
    
    # -----------------------------------------------------------
    # PASO 1: CONFIGURACIÓN DE LA SUITE Y EL OBSERVADOR
    # -----------------------------------------------------------
    # Suite: Es el conjunto de problemas (banco de pruebas). "bbob" es el estándar.
    suite_name = "bbob"
    
    # Observer: Es el "notario". Observa tu algoritmo y guarda los datos automáticamente
    # en la carpeta que le indiquemos.
    output_folder = f"{algorithm_name}_results"
    observer = cocoex.Observer(suite_name, f"result_folder: {output_folder}")
    
    # Creamos la suite filtrando problemas si quisieras (aquí dejamos todos)
    suite = cocoex.Suite(suite_name, "", "")
    
    print(f"🚀 Iniciando Benchmark BBOB para: {algorithm_name}")
    print(f"📂 Los resultados se guardarán en: {output_folder}")

    # -----------------------------------------------------------
    # PASO 2: BUCLE PRINCIPAL (PROBLEMA POR PROBLEMA)
    # -----------------------------------------------------------
    # COCO es un iterador. Nos va a dar uno por uno los problemas:
    # Ej: Sphere 2D, Sphere 5D... Rastrigin 2D... etc.
    for problem in suite:
        
        # Conectamos el "notario" al problema actual.
        # A partir de aquí, cada vez que llames a problem(x), se guarda el dato.
        problem.observe_with(observer) 
        
        # -------------------------------------------------------
        # PASO 3: ADAPTACIÓN DE DATOS (EL "PUENTE")
        # -------------------------------------------------------
        # Extraemos la información del problema que nos da COCO
        dim = problem.dimension
        
        # Los límites vienen como arrays [-5, -5, ...] y [5, 5, ...]
        # Tomamos el primer valor porque tus algoritmos asumen un hipercubo uniforme.
        lower_bound = problem.lower_bounds[0] 
        upper_bound = problem.upper_bounds[0]
        
        # CÁLCULO DEL PRESUPUESTO (BUDGET)
        # BBOB se mide en "Evaluaciones de Función" (FE), no en iteraciones.
        max_evals = dim * budget_multiplier
        
        optimizer = None
        NP = 50 # Población fija para el experimento

        # -------------------------------------------------------
        # PASO 4: INSTANCIACIÓN DEL ALGORITMO
        # -------------------------------------------------------
        if algorithm_name == "MBO":
            # MBO realiza aprox. 2 evaluaciones por individuo por ciclo 
            # (una en migración, una en ajuste).
            # Fórmula: Iteraciones = Total_Evals / (Población * 2)
            iterations = int(max_evals / (NP * 2))
            
            optimizer = MBO(
                NP=NP, 
                p=5/12, 
                num_dim=dim, 
                min_p=lower_bound, 
                max_p=upper_bound, 
                fitness=problem, # <--- LA CLAVE: Pasamos el problema como función fitness
                bar=5/12, 
                peri=1.2, 
                iteration=iterations, 
                S_max=2.0
            )
            
        elif algorithm_name == "PSO":
            # PSO realiza 1 evaluación por partícula por ciclo.
            # Fórmula: Iteraciones = Total_Evals / Población
            iterations = int(max_evals / NP)
            
            optimizer = PSO(
                num_dim=dim, 
                min_p=lower_bound, 
                max_p=upper_bound,
                num_particles=NP, 
                iter=iterations,
                w=0.7, 
                local_weight=1.5, 
                global_weight=1.5,
                fitness=problem # <--- LA CLAVE
            )

        # -------------------------------------------------------
        # PASO 5: EJECUCIÓN
        # -------------------------------------------------------
        # Usamos try/except para que si una función falla, no se detenga todo el benchmark.
        try:
            # Al llamar a run(), tu algoritmo llamará a self.fitness(pos).
            # Como self.fitness es 'problem', COCO recibe la posición, calcula el valor,
            # y el Observer lo anota en el archivo de resultados.
            optimizer.run()
            
        except Exception as e:
            print(f"⚠️ Error en problema {problem.id}: {e}")

    print(f"✅ Benchmark finalizado para {algorithm_name}")

if __name__ == '__main__':
    # Ejecutamos ambos benchmarks.
    # budget_multiplier=20 es un valor bajo para probar rápido (tardará unos minutos).
    # Para el reporte final, los papers suelen usar 1000 o 10000 (tardará horas).
    
    run_experiment("MBO", budget_multiplier=20)
    run_experiment("PSO", budget_multiplier=20)