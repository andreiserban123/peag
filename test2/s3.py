import numpy as np

# Definirea costurilor și a caracteristicilor cursurilor
cost = np.array([1000, 800, 1500])
credite = np.array([5, 3, 8])
ore_studiu = np.array([80, 40, 100])
budget = 10000
max_ore_studiu = 70

######################## CALCUL FUNCTIE FITNESS + FEZ ######################
def calc_credite_medii(a, b, c):
    total_credite = 5 * a + 3 * b + 8 * c
    total_cursuri = a + b + c
    if total_cursuri == 0:
        return 0
    return total_credite / total_cursuri

def calc_ore_studiu_medii(a, b, c):
    total_ore_studiu = 80 * a + 40 * b + 100 * c
    total_cursuri = a + b + c
    if total_cursuri == 0:
        return 0
    return total_ore_studiu / total_cursuri

def fitness(x):
    a, b, c = x
    return calc_credite_medii(a, b, c)

def is_feasible(x):
    a, b, c = x
    cost_total = 1000 * a + 800 * b + 1500 * c
    ore_studiu_medii = calc_ore_studiu_medii(a, b, c)
    return cost_total <= budget and ore_studiu_medii <= max_ore_studiu

########################## GENERAREA UNEI POPULATII ########################
def gen_populatie(dim):
    pop = []
    while len(pop) < dim:
        a = np.random.randint(0, budget // 1000 + 1)
        b = np.random.randint(0, budget // 800 + 1)
        c = np.random.randint(0, budget // 1500 + 1)
        if is_feasible([a, b, c]):
            pop.append([a, b, c, fitness([a, b, c])])
    return np.array(pop)

########################### OPERATOR MUTATIE #########################
def mutatie(x):
    a, b, c = x
    mutatie_idx = np.random.randint(0, 3)
    if mutatie_idx == 0:
        a += np.random.randint(-2, 3)
    elif mutatie_idx == 1:
        b += np.random.randint(-2, 3)
    else:
        c += np.random.randint(-2, 3)
    a = max(0, a)
    b = max(0, b)
    c = max(0, c)
    return [a, b, c]

def mutatie_populatie(pop, probabilitate_m):
    pop_m = pop.copy()
    for i in range(len(pop)):
        if np.random.uniform(0, 1) <= probabilitate_m:
            nou = mutatie(pop[i][:3])
            if is_feasible(nou):
                pop_m[i][:3] = nou
                pop_m[i][3] = fitness(nou)
    return pop_m

########################## OPERATOR RECOMBINARE ########################
def recombinare_uniforma(x1, x2):
    a1, b1, c1 = x1
    a2, b2, c2 = x2
    a = a1 if np.random.uniform(0, 1) > 0.5 else a2
    b = b1 if np.random.uniform(0, 1) > 0.5 else b2
    c = c1 if np.random.uniform(0, 1) > 0.5 else c2
    return [a, b, c]

def recombinare_populatie(pop, probabilitate_crossover):
    copii = pop.copy()
    for i in range(0, len(pop) - 1, 2):
        if np.random.uniform(0, 1) <= probabilitate_crossover:
            c1 = recombinare_uniforma(pop[i][:3], pop[i+1][:3])
            c2 = recombinare_uniforma(pop[i+1][:3], pop[i][:3])
            if is_feasible(c1):
                copii[i][:3] = c1
                copii[i][3] = fitness(c1)
            if is_feasible(c2):
                copii[i+1][:3] = c2
                copii[i+1][3] = fitness(c2)
    return copii

########################## SELECTIE PARENTI SUS ########################
def fps(fitnessuri):
    suma = np.sum(fitnessuri)
    fps = fitnessuri / suma
    qfps = np.cumsum(fps)
    return qfps

def ruleta(pop_initiala):
    dim = len(pop_initiala)
    n = 3
    fitnessuri = pop_initiala[:, -1]
    qfps = fps(fitnessuri)
    parinti = pop_initiala.copy()
    for i in range(dim):
        r = np.random.uniform(0, 1)
        pozitie = np.where(qfps >= r)[0][0]
        parinti[i][:n] = pop_initiala[pozitie][:n]
        parinti[i][n] = fitnessuri[pozitie]
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
def GA(dim, NMAX, pc, pm):
    pop_initiala = gen_populatie(dim)
    istoric_v = [np.max(pop_initiala[:, -1])]
    it = 0
    gata = False
    nrm = 1

    while it < NMAX and not gata:
        parinti = ruleta(pop_initiala)
        pop_copii = recombinare_populatie(parinti, pc)
        pop_copii_mutanti = mutatie_populatie(pop_copii, pm)
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
    individ_max_gene = pop_urmatoare[poz_max][:3]
    individ_max_fitness = maxim

    return np.asarray(individ_max_gene), individ_max_fitness

######################################## RULARE ##############################################
if __name__ == "__main__":
    individ_max_gene, individ_max_fitness = GA(100, 200, 0.8, 0.2)
    print("Gene: ", individ_max_gene)
    print("Fitness: ", individ_max_fitness)
