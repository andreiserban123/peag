import numpy as np

# Citirea matricei de distanțe din fișier
D = np.loadtxt('distant.txt')

# Datele problemei
n = 10  # numărul de insule de vizitat
k = 500  # distanța totală maximă

def calculeaza_distanta(individ):
    distanta_totala = 0
    for i in range(len(individ) - 1):
        distanta_totala += D[individ[i], individ[i+1]]
    distanta_totala += D[individ[-1], individ[0]]  # Întoarcerea la port
    return distanta_totala

def fitness(individ):
    distanta_totala = calculeaza_distanta(individ)
    if distanta_totala <= k:
        distante_insule = sum(D[individ[i], individ[i+1]] for i in range(len(individ) - 1))
        return distante_insule
    else:
        return 0  # Penalizare pentru depășirea distanței maxime

def gen_populatie(dim):
    pop = []
    for _ in range(dim):
        individ = np.random.permutation(n)
        pop.append(np.append(individ, fitness(individ)))
    return np.array(pop)

def mutatie(individ):
    individ = individ[:-1].copy()
    i, j = np.random.randint(0, n, 2)
    individ[i], individ[j] = individ[j], individ[i]
    return np.append(individ, fitness(individ))

def mutatie_populatie(pop, probabilitate_m):
    pop_m = pop.copy()
    for i in range(len(pop)):
        if np.random.uniform(0, 1) <= probabilitate_m:
            pop_m[i] = mutatie(pop[i])
    return pop_m

def pmx_crossover(p1, p2):
    size = len(p1) - 1  # Exclude fitness value
    p1, p2 = p1[:-1].copy(), p2[:-1].copy()
    cxpoint1, cxpoint2 = sorted(np.random.choice(range(size), 2, replace=False))
    temp1, temp2 = p1[cxpoint1:cxpoint2+1], p2[cxpoint1:cxpoint2+1]

    def pmx(parent1, parent2, temp1, temp2):
        for k in range(len(temp1)):
            while temp1[k] in parent1[cxpoint1:cxpoint2+1]:
                index = np.where(parent1 == temp1[k])[0][0]
                temp1[k] = parent2[index]
            while temp2[k] in parent2[cxpoint1:cxpoint2+1]:
                index = np.where(parent2 == temp2[k])[0][0]
                temp2[k] = parent1[index]
        parent1[cxpoint1:cxpoint2+1], parent2[cxpoint1:cxpoint2+1] = temp2, temp1
        return parent1, parent2

    c1, c2 = pmx(p1, p2, temp1, temp2)
    return np.append(c1, fitness(c1)), np.append(c2, fitness(c2))

def recombinare_populatie(pop, probabilitate_crossover):
    copii = pop.copy()
    for i in range(0, len(pop) - 1, 2):
        if np.random.uniform(0, 1) <= probabilitate_crossover:
            p1 = pop[i]
            p2 = pop[i+1]
            copil1, copil2 = pmx_crossover(p1, p2)
            copii[i] = copil1
            copii[i+1] = copil2
    return copii

def fps(fitnessuri):
    fitnessuri = np.array(fitnessuri)
    suma = np.sum(fitnessuri)
    fps = fitnessuri / suma
    qfps = np.cumsum(fps)
    return qfps

def ruleta(pop_initiala):
    dim = len(pop_initiala)
    fitnessuri = [individ[-1] for individ in pop_initiala]
    qfps = fps(fitnessuri)
    parinti = pop_initiala.copy()
    for i in range(dim):
        r = np.random.uniform(0, 1)
        pozitie = np.where(qfps >= r)[0][0]
        parinti[i] = pop_initiala[pozitie]
    return parinti

def elitism(pop_curenta, pop_mutanta):
    dim = len(pop_curenta)
    pop_urmatoare = pop_mutanta.copy()
    max_curent = max([individ[-1] for individ in pop_curenta])
    max_mutant = max([individ[-1] for individ in pop_mutanta])
    if max_curent > max_mutant:
        poz = [i for i, individ in enumerate(pop_curenta) if individ[-1] == max_curent][0]
        ir = np.random.randint(dim)
        pop_urmatoare[ir] = pop_curenta[poz]
    return pop_urmatoare

def GA(dim, NMAX, pc, pm):
    pop_initiala = gen_populatie(dim)
    istoric_v = [max([individ[-1] for individ in pop_initiala])]
    it = 0
    gata = False
    nrm = 1

    while it < NMAX and not gata:
        parinti = ruleta(pop_initiala)
        pop_copii = recombinare_populatie(parinti, pc)
        pop_copii_mutanti = mutatie_populatie(pop_copii, pm)
        pop_urmatoare = elitism(pop_initiala, pop_copii_mutanti)
        minim = min([individ[-1] for individ in pop_urmatoare])
        maxim = max([individ[-1] for individ in pop_urmatoare])
        if maxim == istoric_v[it]:
            nrm += 1
        else:
            nrm = 0
        if maxim == minim or nrm == int(NMAX / 4):
            gata = True
        else:
            it += 1
        istoric_v.append(maxim)
        pop_initiala = pop_urmatoare.copy()

    poz_max = [i for i, individ in enumerate(pop_urmatoare) if individ[-1] == maxim][0]
    individ_max_gene = pop_urmatoare[poz_max][:-1]
    individ_max_fitness = maxim

    return individ_max_gene, individ_max_fitness

if __name__ == "__main__":
    individ_max_gene, individ_max_fitness = GA(100, 200, 0.8, 0.2)
    print("Gene: ", individ_max_gene)
    print("Fitness: ", individ_max_fitness)
