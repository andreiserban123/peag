import numpy as np

# Capacitățile navelor - exemplu hardcodat
capacities = np.asarray([30, 10, 32, 100, 53])

# Datele problemei
n = len(capacities)  # numărul de nave
m = 5  # numărul de exploatări

########################## FUNCTIE FITNESS + FEZABILITATE ########################
def calculeaza_capacitati(individ):
    capacitati_exploatari = np.zeros(m)
    for i in range(n):
        capacitati_exploatari[int(individ[i])] += capacities[i]
    return capacitati_exploatari

def fitness(individ):
    capacitati = calculeaza_capacitati(individ)
    deviatia_standard = np.std(capacitati)
    return 1 / (1 + deviatia_standard)  # Fitness mai mare pentru deviatia standard mai mică

########################## GENERAREA UNEI POPULATII ########################
def gen_populatie(dim):
    pop = []
    for _ in range(dim):
        individ = np.random.randint(0, m, n)
        pop.append(np.append(individ, fitness(individ)))
    return np.array(pop)

########################### OPERATOR MUTATIE #########################
def mutatie(individ):
    individ = individ[:-1].astype(int).copy()
    i = np.random.randint(0, n)
    individ[i] = np.random.randint(0, m)
    return np.append(individ, fitness(individ))

def mutatie_populatie(pop, probabilitate_m):
    pop_m = pop.copy()
    for i in range(len(pop)):
        if np.random.uniform(0, 1) <= probabilitate_m:
            pop_m[i] = mutatie(pop[i])
    return pop_m

########################## OPERATOR RECOMBINARE ########################
def recombinare_uniforma(p1, p2):
    mask = np.random.randint(0, 2, size=n)
    copil = p1[:-1].astype(int).copy()
    copil[mask == 1] = p2[:-1].astype(int)[mask == 1]
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

########################## SELECTIE PARENTI SUS ########################
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

########################## ELITISM ########################
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

##################################### APLICARE GA FINAL ###################################
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

######################################## RULARE ##############################################
if __name__ == "__main__":
    individ_max_gene, individ_max_fitness = GA(100, 200, 0.8, 0.2)
    print("Gene: ", individ_max_gene)
    print("Fitness: ", individ_max_fitness)
