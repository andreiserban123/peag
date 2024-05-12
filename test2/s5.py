import numpy as np

greutati = np.genfromtxt("container_weights.txt", dtype=int)


def ok(x, greutati, n):
    x = list(x)
    g = np.zeros(n)
    for i in range(n):
        for j in range(len(x)):
            if x[j] == i:
                g[i] += greutati[j]

    dev = np.std(g)
    return 1 / (dev + 1)


def crossover_unipunct(x1, x2, n):
    p = np.random.randint(0, n)
    c1 = x1.copy()
    c2 = x2.copy()
    c1[0:p] = x1[0:p]
    c1[p:] = x2[p:n]
    c2[0:p] = x2[0:p]
    c2[p:n] = x1[p:n]
    return c1, c2


def gen(dim, n):
    pop = []

    m = len(greutati)
    for i in range(dim):
        x = np.random.randint(0, n, m)
        fitness = ok(x, greutati, n)
        x = list(x)
        x.append(fitness)
        pop.append(x)
    return pop


def crossover_populatie(pop, dim, n, probabilitate_crossover):
    copii = pop.copy()
    copii = list(copii)
    for i in range(0, dim - 1, 2):
        # selecteaza parintii
        x1 = pop[i][0:n].copy()
        x2 = pop[i + 1][0:n].copy()
        r = np.random.uniform(0, 1)
        if r <= probabilitate_crossover:
            c1, c2 = crossover_unipunct(x1, x2, n)
            fitness = ok(c1, greutati, n)
            c1 = list(c1)
            c1 = c1 + [fitness]
            copii[i] = c1.copy()
            fitness = ok(c2, greutati, n)
            c2 = list(c2)
            c2 = c2 + [fitness]
            copii[i + 1] = c2.copy()
    return copii


def m_uniforma(x, n):
    gena = np.random.randint(0, n)
    x[gena] = np.random.randint(0, n)
    return x


def mutatie_populatie(pop, dim, n, probabilitate_mutatie):
    pop = np.asarray(pop)
    pop_m = pop.copy()
    for i in range(dim):
        x = pop[i][:n].copy()
        r = np.random.uniform(0, 1)
        if r <= probabilitate_mutatie:
            x = m_uniforma(x, n)
            val = ok(x, greutati, n)
            x = list(x)
            x = x + [val]
            pop_m[i] = x.copy
    return pop_m


def turnir(pop, k):
    pop_turnir = []
    for i in range(k):
        index = np.random.randint(0, len(pop))
        pop_turnir.append(pop[index])
    fitnesses = []
    for i in range(k):
        fitnesses.append(pop_turnir[i][-1])
    index_castigator = np.argmax(fitnesses)
    return pop_turnir[index_castigator]


def selectia_parintilor(pop, k):
    pop_parinti = []
    for i in range(len(pop)):
        indiv_castigator = turnir(pop, k)
        pop_parinti.append(indiv_castigator)
    return pop_parinti


def elitism(pop_curenta, pop_mutanta, dim, n):
    pop_curenta = np.asarray(pop_curenta)
    pop_mutanta = np.asarray(pop_mutanta)
    pop_urmatoare = np.copy(pop_mutanta)
    max_curent = np.max(pop_curenta[:, -1])
    max_mutant = np.max(pop_mutanta[:, -1])
    if max_curent > max_mutant:
        poz = np.where(pop_curenta[:, -1] == max_curent)
        imax = poz[0][0]
        ir = np.random.randint(dim)
        pop_urmatoare[ir, 0:n] = pop_curenta[imax, 0:n].copy()
        pop_urmatoare[ir, n] = max_curent
    return pop_urmatoare


def GA(dim, n, NMAX, pc, pm):
    pop_initiala = gen(dim, n)
    pop_initiala = np.asarray(pop_initiala)
    istoric_v = [np.max(pop_initiala[:, -1])]
    it = 0
    gata = False
    nrm = 1
    while it < NMAX and not gata:
        parinti = selectia_parintilor(pop_initiala, 8)
        parinti = np.asarray(parinti)
        pop_copii = crossover_populatie(parinti, dim, n, pc)
        pop_copii_mutanti = mutatie_populatie(pop_copii, dim, n, pm)
        pop_urmatoare = elitism(pop_initiala, pop_copii_mutanti, dim, n)
        minim = np.min(pop_urmatoare[:, -1])
        maxim = np.max(pop_urmatoare[:, -1])
        if maxim == istoric_v[it]:
            nrm = nrm + 1
        else:
            nrm = 0
        if maxim == minim or nrm == int(NMAX / 4):
            gata = True
        else:
            it = it + 1
        istoric_v.append(maxim)
        pop_initiala = pop_urmatoare.copy()
    poz_max = np.where(pop_urmatoare[:, -1] == maxim)
    individ_max = pop_urmatoare[poz_max[0][0], 0:]
    fitness_max = maxim
    return np.asarray(individ_max), fitness_max


# Setările pentru GA
dim = 100  # Dimensiunea populației
n = 5  # Numărul de insule (sau vagoane)
NMAX = 50  # Numărul maxim de generații
pc = 0.8  # Probabilitatea de crossover
pm = 0.1  # Probabilitatea de mutație

best_solution, fit = GA(dim, n, NMAX, pc, pm)
print("Best solution found:", best_solution)
print("Best fitness:", fit)
