import numpy as np

# Datele problemei
costuri = np.array([50, 70, 90, 60, 70, 100])

capacitate_fabrica = [120, 140]
necesar_depozite = [100, 60, 80]

########################## FUNCTIE FITNESS + FEZABILITATE ########################
def calculeaza_cost(X):
    return np.sum(X * costuri)

def is_feasible(X):
    # Verificăm constrângerile capacităților fabricilor
    if X[0] + X[1] + X[2] > capacitate_fabrica[0]:  # București
        return False
    if X[3] + X[4] + X[5] > capacitate_fabrica[1]:  # Craiova
        return False
    # Verificăm constrângerile necesarului depozitelor
    if X[0] + X[3] != necesar_depozite[0]:  # Ploiești
        return False
    if X[1] + X[4] != necesar_depozite[1]:  # Pitești
        return False
    if X[2] + X[5] != necesar_depozite[2]:  # Cluj
        return False
    return True

def fitness(X):
    if not is_feasible(X):
        return float('inf')
    return calculeaza_cost(X)

########################## GENERAREA UNEI POPULATII ########################
def gen_populatie(dim):
    pop = []
    for _ in range(dim):
        X = np.zeros(6, dtype=int)
        X[0] = np.random.randint(0, min(capacitate_fabrica[0], necesar_depozite[0]) + 1)
        X[3] = necesar_depozite[0] - X[0]

        X[1] = np.random.randint(0, min(capacitate_fabrica[0] - X[0], necesar_depozite[1]) + 1)
        X[4] = necesar_depozite[1] - X[1]

        X[2] = np.random.randint(0, min(capacitate_fabrica[0] - X[0] - X[1], necesar_depozite[2]) + 1)
        X[5] = necesar_depozite[2] - X[2]

        if is_feasible(X):
            pop.append((X, fitness(X)))
    return pop

########################### OPERATOR MUTATIE #########################
def mutatie(X):
    i = np.random.randint(0, 6)
    cantitate = np.random.randint(-5, 6)
    X[i] = max(0, X[i] + cantitate)
    return X

def mutatie_populatie(pop, probabilitate_m):
    pop_m = pop.copy()
    for i in range(len(pop)):
        if np.random.uniform(0, 1) <= probabilitate_m:
            X = pop[i][0].copy()
            X = mutatie(X)
            if is_feasible(X):
                pop_m[i] = (X, fitness(X))
    return pop_m

########################## OPERATOR RECOMBINARE ########################
def recombinare_uniforma(X1, X2):
    X_new = X1.copy()
    mask = np.random.randint(0, 2, size=6)
    X_new[mask == 1] = X2[mask == 1]
    return X_new

def recombinare_populatie(pop, probabilitate_crossover):
    copii = pop.copy()
    for i in range(0, len(pop) - 1, 2):
        if np.random.uniform(0, 1) <= probabilitate_crossover:
            X1 = pop[i][0]
            X2 = pop[i+1][0]
            X_new1 = recombinare_uniforma(X1, X2)
            X_new2 = recombinare_uniforma(X2, X1)
            if is_feasible(X_new1):
                copii[i] = (X_new1, fitness(X_new1))
            if is_feasible(X_new2):
                copii[i+1] = (X_new2, fitness(X_new2))
    return copii

########################## SELECTIE PARENTI SUS ########################
def fps(fitnessuri):
    fitnessuri = np.array(fitnessuri)
    suma = np.sum(fitnessuri)
    fps = fitnessuri / suma
    qfps = np.cumsum(fps)
    return qfps

def ruleta(pop_initiala):
    dim = len(pop_initiala)
    fitnessuri = [individ[1] for individ in pop_initiala]
    qfps = fps(fitnessuri)
    parinti = pop_initiala.copy()
    for i in range(dim):
        r = np.random.uniform(0, 1)
        pozitie = np.where(qfps >= r)[0][0]
        parinti[i] = pop_initiala[pozitie]
    return parinti

########################## ELITISM ########################
def elitism(pop_curenta, pop_mutanta):
    dim = len(pop_curenta)
    pop_urmatoare = pop_mutanta.copy()
    max_curent = min([individ[1] for individ in pop_curenta])
    max_mutant = min([individ[1] for individ in pop_mutanta])
    if max_curent < max_mutant:
        poz = [i for i, individ in enumerate(pop_curenta) if individ[1] == max_curent][0]
        ir = np.random.randint(dim)
        pop_urmatoare[ir] = pop_curenta[poz]
    return pop_urmatoare

##################################### APLICARE GA FINAL ###################################
def GA(dim, NMAX, pc, pm):
    pop_initiala = gen_populatie(dim)
    istoric_v = [min([individ[1] for individ in pop_initiala])]
    it = 0
    gata = False
    nrm = 1

    while it < NMAX and not gata:
        parinti = ruleta(pop_initiala)
        pop_copii = recombinare_populatie(parinti, pc)
        pop_copii_mutanti = mutatie_populatie(pop_copii, pm)
        pop_urmatoare = elitism(pop_initiala, pop_copii_mutanti)
        minim = min([individ[1] for individ in pop_urmatoare])
        maxim = max([individ[1] for individ in pop_urmatoare])
        if minim == istoric_v[it]:
            nrm += 1
        else:
            nrm = 0
        if minim == maxim or nrm == int(NMAX / 4):
            gata = True
        else:
            it += 1
        istoric_v.append(minim)
        pop_initiala = pop_urmatoare.copy()

    poz_max = [i for i, individ in enumerate(pop_urmatoare) if individ[1] == minim][0]
    individ_max_gene = pop_urmatoare[poz_max][0]
    individ_max_fitness = minim

    return individ_max_gene, individ_max_fitness

######################################## RULARE ##############################################
if __name__ == "__main__":
    individ_max_gene, individ_max_fitness = GA(100, 200, 0.8, 0.2)
    print("Gene: ", individ_max_gene)
    print("Fitness: ", individ_max_fitness)
