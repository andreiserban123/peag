import numpy


def ok(x, n, cmax, c, v, t):
    cost = 0
    fitness = 0
    tcas = 0
    sum = 0
    for i in range(n):
        cost += x[i] * c[i]
        fitness += x[i] * v[i]
        tcas += x[i] * t[i]
        sum += x[i]
    tcas = tcas / sum
    fitness = fitness / sum
    if tcas < 40:
        return False, fitness
    return cost <= cmax, fitness


def gen(cost_max, dim):
    pop = []
    cmax = cost_max
    c = numpy.genfromtxt("cost_aeronave.txt")
    v = numpy.genfromtxt("autonomie.txt")
    t = numpy.genfromtxt("TCAS.txt")
    n = len(c)
    for i in range(dim):
        flag = False
        while not flag:
            x = numpy.random.randint(0, int(cmax / 50), n, dtype=int)
            flag, fitness = ok(x, n, cmax, c, v, t)
        x = list(x)
        x.append(fitness)
        pop.append(x)
    return pop, dim, n, c, v, t, cmax


def fps(fitnessuri, dim):
    fps = numpy.zeros(dim)
    suma = numpy.sum(fitnessuri)
    for i in range(dim):
        fps[i] = fitnessuri[i] / suma
    qfps = fps.copy()
    for i in range(1, dim):
        qfps[i] = qfps[i - 1] + fps[i]
    return qfps  # return array()


def ruleta(pop_initiala, dim, n):
    pop_initiala = numpy.asarray(pop_initiala)
    parinti = pop_initiala.copy()
    fitnessuri = numpy.zeros(dim, dtype="float")
    for i in range(dim):
        fitnessuri[i] = pop_initiala[i][n]
    qfps = fps(fitnessuri, dim)
    for i in range(dim):
        r = numpy.random.uniform(0, 1)
        pozitie = numpy.where(qfps >= r)
        index_buzunar_ruleta = pozitie[0][0]
        parinti[i][0:n] = pop_initiala[index_buzunar_ruleta][0:n]
        parinti[i][n] = fitnessuri[index_buzunar_ruleta]
    return parinti


def crossover_unipunct(x1, x2, n):
    p = numpy.random.randint(0, n)
    c1 = x1.copy()
    c2 = x2.copy()
    c1[0:p] = x1[0:p]
    c1[p:] = x2[p:n]
    c2[0:p] = x2[0:p]
    c2[p:n] = x1[p:n]
    return c1, c2


def crossover_populatie(pop, dim, n, c, v, t, cmax, probabilitate_crossover):
    copii = pop.copy()
    for i in range(0, dim - 1, 2):
        # selecteaza parintii
        x1 = pop[i][0:n].copy()
        x2 = pop[i + 1][0:n].copy()
        r = numpy.random.uniform(0, 1)
        if r <= probabilitate_crossover:
            c1, c2 = crossover_unipunct(x1, x2, n)
            flag, fitness = ok(c1, n, cmax, c, v, t)
            if flag:
                c1 = list(c1)
                c1 = c1 + [fitness]
                copii[i] = c1.copy()
            flag, fitness = ok(c2, n, cmax, c, v, t)
            if flag:
                c2 = list(c2)
                c2 = c2 + [fitness]
                copii[i + 1] = c2.copy()
    return copii


def m_uniforma(gena, cmax):
    gena_mutanta = numpy.random.randint(0, int(cmax / 50))
    while gena_mutanta == gena:
        gena_mutanta = numpy.random.randint(0, int(cmax / 50))
    return gena_mutanta


def mutatie_populatie(pop, dim, n, c, v, t, cost_max, probabilitate_mutatie):
    pop_m = pop.copy()
    for i in range(dim):
        x = pop[i][:n].copy()
        for j in range(n):
            r = numpy.random.uniform(0, 1)
            if r <= probabilitate_mutatie:
                x[j] = m_uniforma(x[j], cost_max)
        fez, val = ok(x, n, cost_max, c, v, t)
        if fez:
            x = list(x)
            x = x + [val]
            pop_m[i] = x.copy()
    return pop_m


def elitism(pop_curenta, pop_mutanta, dim, n):
    pop_curenta = numpy.asarray(pop_curenta)
    pop_mutanta = numpy.asarray(pop_mutanta)
    pop_urmatoare = numpy.copy(pop_mutanta)
    max_curent = numpy.max(pop_curenta[:, -1])
    max_mutant = numpy.max(pop_mutanta[:, -1])
    if max_curent > max_mutant:
        poz = numpy.where(pop_curenta[:, -1] == max_curent)
        imax = poz[0][0]
        ir = numpy.random.randint(dim)
        pop_urmatoare[ir, 0:n] = pop_curenta[imax, 0:n].copy()
        pop_urmatoare[ir, n] = max_curent
    return pop_urmatoare


def GA(cost_max, dim, NMAX, pc, pm):
    pop_initiala, dim, n, c, v, t, cost_max = gen(cost_max, dim)
    pop_initiala = numpy.asarray(pop_initiala)
    # pastram cel mai bun cost din populatia curenta la fiecare moment al evolutiei
    istoric_v = [numpy.max(pop_initiala[:, -1])]
    it = 0
    gata = False
    nrm = 1
    while it < NMAX and not gata:
        parinti = ruleta(pop_initiala, dim, n)
        pop_copii = crossover_populatie(parinti, dim, n, c, v, t, cost_max, pc)
        pop_copii_mutanti = mutatie_populatie(pop_copii, dim, n, c, v, t, cost_max, pm)
        pop_urmatoare = elitism(pop_initiala, pop_copii_mutanti, dim, n)
        minim = numpy.min(pop_urmatoare[:, -1])
        maxim = numpy.max(pop_urmatoare[:, -1])
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

    poz_max = numpy.where(pop_urmatoare[:, -1] == maxim)
    individ_max_gene = pop_urmatoare[poz_max[0][0], 0:]
    individ_max_fitness = maxim
    return numpy.asarray(individ_max_gene), individ_max_fitness


if __name__ == '__main__':
    individ_max_gene, individ_max_fitness = GA(5000, 100, 200, 0.8, 0.1)
    print(individ_max_fitness)
    print(individ_max_gene)
