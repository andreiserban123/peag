import numpy as np

# Datele problemei
buget = 5000
costuri = np.array([100, 60, 50])
autonomii = np.array([6000, 4200, 2800])
raze = np.array([30, 48, 32])

def calculeaza_cost_total(individ):
    return np.dot(individ, costuri)

def calculeaza_autonomie_medie(individ):
    total_aeronave = np.sum(individ)
    if total_aeronave == 0:
        return 0
    return np.dot(individ, autonomii) / total_aeronave

def calculeaza_raza_medie(individ):
    total_aeronave = np.sum(individ)
    if total_aeronave == 0:
        return 0
    return np.dot(individ, raze) / total_aeronave

def fitness(individ):
    cost_total = calculeaza_cost_total(individ)
    if cost_total > buget:
        return 0
    raza_medie = calculeaza_raza_medie(individ)
    if raza_medie < 40:
        return 0
    autonomie_medie = calculeaza_autonomie_medie(individ)
    return autonomie_medie

def gen_populatie(dim):
    pop = []
    for _ in range(dim):
        individ = np.random.randint(0, buget // np.min(costuri), 3)
        pop.append(np.append(individ, fitness(individ)))
    return np.array(pop)

def mutatie(individ):
    individ = individ[:-1].copy()
    i = np.random.randint(0, 3)
    individ[i] = np.random.randint(0, buget // costuri[i])
    return np.append(individ, fitness(individ))

def mutatie_populatie(pop, probabilitate_m):
    pop_m = pop.copy()
    for i in range(len(pop)):
        if np.random.uniform(0, 1) <= probabilitate_m:
            pop_m[i] = mutatie(pop[i])
    return pop_m

def recombinare_uniforma(p1, p2):
    mask = np.random.randint(0, 2, size=3)
    copil = p1[:-1].copy()
    copil[mask == 1] = p2[:-1][mask == 1]
    return np.append(copil, fitness(copil))

def recombinare_populatie(pop, probabilitate_crossover):
    copii = pop.copy()
    for i in range(0, len(pop) - 1, 2):
        if np.random.uniform(0, 1) <= probabilitate_crossover:
            p1 = pop[i]
            p2 = pop[i+1]
            copil1 = recombinare_uniforma(p1, p2)
            copil2 = recombinare_uniforma(p2, p1)
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
