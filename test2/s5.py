import numpy as np

# Hardcodarea maselor containerelor
def citeste_mase():
    mase = [120, 150, 200, 100, 180, 140, 130, 170, 160, 190, 210, 125, 135, 145, 155]
    return mase

# Definirea funcțiilor de fitness și de verificare a fezabilității
def calculeaza_mase_vagoane(allocation, mase, n):
    mase_vagoane = [0] * n
    for i in range(len(allocation)):
        mase_vagoane[int(allocation[i])] += mase[i]
    return mase_vagoane

def fitness(allocation, mase, n):
    mase_vagoane = calculeaza_mase_vagoane(allocation, mase, n)
    return 1 / (1 + (max(mase_vagoane) - min(mase_vagoane)))

########################## GENERAREA UNEI POPULATII ########################
def gen_populatie(dim, mase, n):
    pop = []
    m = len(mase)
    for _ in range(dim):
        allocation = np.random.randint(0, n, m).tolist()
        pop.append(allocation + [fitness(allocation, mase, n)])
    return np.array(pop)

########################### OPERATOR MUTATIE #########################
def mutatie(allocation, n):
    m = len(allocation)
    container_idx = np.random.randint(0, m)
    new_vagon = np.random.randint(0, n)
    allocation[container_idx] = new_vagon
    return allocation

def mutatie_populatie(pop, mase, n, probabilitate_m):
    pop_m = pop.copy()
    for i in range(len(pop)):
        if np.random.uniform(0, 1) <= probabilitate_m:
            nou = mutatie(pop[i][:len(mase)].copy(), n)
            pop_m[i][:len(mase)] = nou
            pop_m[i][-1] = fitness(nou, mase, n)
    return pop_m

########################## OPERATOR RECOMBINARE ########################
def recombinare_uniforma(x1, x2):
    m = len(x1)
    child = [x1[i] if np.random.uniform(0, 1) > 0.5 else x2[i] for i in range(m)]
    return child

def recombinare_populatie(pop, mase, n, probabilitate_crossover):
    copii = pop.copy()
    for i in range(0, len(pop) - 1, 2):
        if np.random.uniform(0, 1) <= probabilitate_crossover:
            c1 = recombinare_uniforma(pop[i][:len(mase)], pop[i+1][:len(mase)])
            c2 = recombinare_uniforma(pop[i+1][:len(mase)], pop[i][:len(mase)])
            copii[i][:len(mase)] = c1
            copii[i][-1] = fitness(c1, mase, n)
            copii[i+1][:len(mase)] = c2
            copii[i+1][-1] = fitness(c2, mase, n)
    return copii

########################## SELECTIE PARENTI SUS ########################
def fps(fitnessuri):
    suma = np.sum(fitnessuri)
    fps = fitnessuri / suma
    qfps = np.cumsum(fps)
    return qfps

def ruleta(pop_initiala):
    dim = len(pop_initiala)
    m = len(pop_initiala[0]) - 1
    fitnessuri = pop_initiala[:, -1]
    qfps = fps(fitnessuri)
    parinti = pop_initiala.copy()
    for i in range(dim):
        r = np.random.uniform(0, 1)
        pozitie = np.where(qfps >= r)[0][0]
        parinti[i][:m] = pop_initiala[pozitie][:m]
        parinti[i][m] = fitnessuri[pozitie]
    return parinti

########################## ELITISM ########################
def elitism(pop_curenta, pop_mutanta):
    dim = len(pop_curenta)
    pop_urmatoare = pop_mutanta.copy()
    max_curent = np.max(pop_curenta[:, -1])
    max_mutant = np.max(pop_mutanta[:, -1])
    if max_curent > max_mutant:
        poz = np.where(pop_curenta[:, -1] == max_curent)[0][0]
        ir = np.random.randint(dim)
        pop_urmatoare[ir] = pop_curenta[poz]
    return pop_urmatoare

##################################### APLICARE GA FINAL ###################################
def GA(dim, NMAX, pc, pm, n):
    mase = citeste_mase()
    pop_initiala = gen_populatie(dim, mase, n)
    istoric_v = [np.max(pop_initiala[:, -1])]
    it = 0
    gata = False
    nrm = 1

    while it < NMAX and not gata:
        parinti = ruleta(pop_initiala)
        pop_copii = recombinare_populatie(parinti, mase, n, pc)
        pop_copii_mutanti = mutatie_populatie(pop_copii, mase, n, pm)
        pop_urmatoare = elitism(pop_initiala, pop_copii_mutanti)
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
    individ_max_gene = pop_urmatoare[poz_max][:len(mase)]
    individ_max_fitness = maxim

    return np.asarray(individ_max_gene), individ_max_fitness

######################################## RULARE ##############################################
if __name__ == "__main__":
    individ_max_gene, individ_max_fitness = GA(100, 200, 0.8, 0.2, 10)
    print("Gene: ", individ_max_gene)
    print("Fitness: ", individ_max_fitness)
