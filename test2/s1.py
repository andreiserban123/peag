import numpy as np

######################## CALCUL FUNCTIE FITNESS + FEZ ######################
def nr_conflicte(x, n, conflict_matrix):
    conflicts = 0
    for i in range(n):
        conflicts += conflict_matrix[x[i]][x[(i+1) % n]]
    return conflicts

def fitness(x, n, conflict_matrix):
    return 1 / (1 + nr_conflicte(x, n, conflict_matrix))

########################## GENERAREA UNEI POPULATII ########################
def gen(dim, conflict_matrix):
    n = len(conflict_matrix)
    pop = []
    for _ in range(dim):
        x = np.random.permutation(n)
        x = list(x)
        x.append(fitness(x, n, conflict_matrix))
        pop.append(x)
    return pop, dim, n, conflict_matrix

########################### OPERATOR MUTATIE #########################
def mutatie_inversiune(x, n):
    poz = np.random.randint(0, n, 2)
    while poz[0] == poz[1]:
        poz = np.random.randint(0, n, 2)
    p1, p2 = min(poz), max(poz)
    x_mutant = x[:p1] + x[p1:p2+1][::-1] + x[p2+1:]
    return x_mutant

def mutatie_populatie(pop, dim, n, conflict_matrix, probabilitate_m):
    pop_m = pop.copy()
    for i in range(dim):
        x = pop[i][:n].copy()
        r = np.random.uniform(0, 1)
        if r <= probabilitate_m:
            x = mutatie_inversiune(x, n)
            x.append(fitness(x, n, conflict_matrix))
            pop_m[i] = x.copy()
    return pop_m

########################## OPERATOR RECOMBINARE PMX ########################
def PMX(x1, x2, n):
    p1, p2 = sorted(np.random.choice(n, 2, replace=False))
    c1, c2 = [-1]*n, [-1]*n
    c1[p1:p2+1], c2[p1:p2+1] = x1[p1:p2+1], x2[p1:p2+1]

    for i in range(p1, p2+1):
        if x2[i] not in c1:
            val = x2[i]
            while val in c1:
                idx = x1.index(val)
                val = x2[idx]
            c1[c1.index(-1)] = val

        if x1[i] not in c2:
            val = x1[i]
            while val in c2:
                idx = x2.index(val)
                val = x1[idx]
            c2[c2.index(-1)] = val

    for i in range(n):
        if c1[i] == -1:
            c1[i] = x2[i]
        if c2[i] == -1:
            c2[i] = x1[i]

    return c1, c2

def crossover_populatie(pop, dim, n, conflict_matrix, probabilitate_crossover):
    copii = pop.copy()
    for i in range(0, dim - 1, 2):
        x1 = pop[i][:n]
        x2 = pop[i + 1][:n]
        r = np.random.uniform(0, 1)
        if r <= probabilitate_crossover:
            c1, c2 = PMX(x1, x2, n)
            c1.append(fitness(c1, n, conflict_matrix))
            c2.append(fitness(c2, n, conflict_matrix))
            copii[i], copii[i + 1] = c1.copy(), c2.copy()
    return copii

########################## SELECTIE PARENTI SUS ########################
def fps(fitnessuri, dim):
    fps = np.zeros(dim)
    suma = np.sum(fitnessuri)
    for i in range(dim):
        fps[i] = fitnessuri[i] / suma
    qfps = np.cumsum(fps)
    return qfps

def ruleta(pop_initiala, dim, n):
    pop_initiala = np.asarray(pop_initiala)
    parinti = pop_initiala.copy()
    fitnessuri = np.array([individ[n] for individ in pop_initiala])
    qfps = fps(fitnessuri, dim)
    for i in range(dim):
        r = np.random.uniform(0, 1)
        pozitie = np.where(qfps >= r)[0][0]
        parinti[i][:n] = pop_initiala[pozitie][:n]
        parinti[i][n] = fitnessuri[pozitie]
    return parinti

########################## ELITISM ########################
def elitism(pop_curenta, pop_mutanta, dim, n):
    pop_curenta = np.asarray(pop_curenta)
    pop_mutanta = np.asarray(pop_mutanta)
    pop_urmatoare = np.copy(pop_mutanta)

    max_curent = np.max(pop_curenta[:, -1])
    max_mutant = np.max(pop_mutanta[:, -1])

    if max_curent > max_mutant:
        poz = np.where(pop_curenta[:, -1] == max_curent)[0][0]
        ir = np.random.randint(dim)
        pop_urmatoare[ir][:n] = pop_curenta[poz][:n].copy()
        pop_urmatoare[ir][n] = max_curent
    return pop_urmatoare

##################################### APLICARE GA FINAL ###################################
def GA(conflict_matrix, dim, NMAX, pc, pm):
    pop_initiala, dim, n, conflict_matrix = gen(dim, conflict_matrix)
    pop_initiala = np.asarray(pop_initiala)
    istoric_v = [np.max(pop_initiala[:, -1])]

    it = 0
    gata = False
    nrm = 1

    while it < NMAX and not gata:
        parinti = ruleta(pop_initiala, dim, n)
        pop_copii = crossover_populatie(parinti, dim, n, conflict_matrix, pc)
        pop_copii_mutanti = mutatie_populatie(pop_copii, dim, n, conflict_matrix, pm)
        pop_urmatoare = elitism(pop_initiala, pop_copii_mutanti, dim, n)
        minim = np.min(pop_urmatoare[:, -1])
        maxim = np.max(pop_urmatoare[:, -1])
        if maxim == istoric_v[it]:
            nrm += 1
        else:
            nrm = 0
        if maxim == minim or nrm == int(NMAX / 4):
            gata = True
        else:
            it += 1
        istoric_v.append(np.max(pop_urmatoare[:, -1]))
        pop_initiala = pop_urmatoare.copy()

    poz_max = np.where(pop_urmatoare[:, -1] == maxim)[0][0]
    individ_max_gene = pop_urmatoare[poz_max][:n]
    individ_max_fitness = maxim

    return np.asarray(individ_max_gene), individ_max_fitness

######################################## RULARE ##############################################
if __name__ == "__main__":
    conflict_matrix = np.array([
        [0, 1, 0, 0, 1],
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1],
        [1, 0, 0, 1, 0]
    ])
    individ_max_gene, individ_max_fitness = GA(conflict_matrix, 10, 100, 0.8, 0.2)
    print("Gene: ", np.asarray(individ_max_gene))
    print("Fitness: ", individ_max_fitness)
