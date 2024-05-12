import numpy

conflicte = numpy.genfromtxt("s1.txt", dtype=int)


def ok(x, conflicte):
    x = numpy.asarray(x)
    x = list(x.astype(int))
    f = 0
    for i in range(len(x) - 1):
        f += conflicte[x[i]][x[i + 1]]
    return 1 / (f + 1)


def gen(dim, conflicte):
    pop = []
    conflicte = list(conflicte)
    for i in range(dim):
        x = numpy.random.permutation(len(conflicte))
        x = list(x)
        fitness = ok(x, conflicte)
        x += [fitness]
        pop.append(x)
    return pop


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


def PMX(x1, x2, n, p1, p2):
    # initializare copil - un vector cu toate elementele -1 - valori care s=sa nu fie in 0,...,n-1
    c = -numpy.ones(n, dtype=int)
    # copiaza secventa comuna in copilul c
    c[p1:p2 + 1] = x1[p1:p2 + 1]
    # analiza secventei comune - in permutarea y
    for i in range(p1, p2 + 1):
        # plasarea alelei a
        a = x2[i]
        if a not in c:
            curent = i
            plasat = False
            while not plasat:
                b = x1[curent]
                # poz=pozitia in care se afla b in y
                [poz] = [j for j in range(n) if x2[j] == b]
                if c[poz] == -1:
                    c[poz] = a
                    plasat = True
                else:
                    curent = poz
    # z= vectorul alelelor din y inca necopiate in c
    z = [x2[i] for i in range(n) if x2[i] not in c]
    # poz - vectorul pozitiilor libere in c - cele cu vaori -1
    poz = [i for i in range(n) if c[i] == -1]
    # copierea alelelor inca necopiate din y in c
    m = len(poz)
    for i in range(m):
        c[poz[i]] = z[i]
    return c


def crossover_PMX(x1, x2, n):
    # generarea secventei de crossover
    poz = numpy.random.randint(0, n, 2)
    while poz[0] == poz[1]:
        poz = numpy.random.randint(0, n, 2)
    p1 = numpy.min(poz)
    p2 = numpy.max(poz)
    c1 = PMX(x2, x2, n, p1, p2)
    c2 = PMX(x2, x1, n, p1, p2)
    return c1, c2


def crossover_populatie(pop, dim, n, pc):
    copii = pop.copy()
    # initializeaza populatia de copii, po, cu matricea cu elementele 0
    for i in range(0, dim - 1, 2):
        # selecteaza parintii
        x = pop[i]
        y = pop[i + 1]

        r = numpy.random.uniform(0, 1)
        if r <= pc:
            # crossover x cu y - PMX
            c1, c2 = crossover_PMX(x[:n], y[:n], n)

            v1 = ok(c1, conflicte)
            v2 = ok(c2, conflicte)

            c1 = list(c1)
            c2 = list(c2)

            c1 = c1 + [v1]
            c2 = c2 + [v2]

            copii[i] = c1.copy()
            copii[i + 1] = c2.copy()
    return copii


def m_perm_interschimbare(x, n):
    poz = numpy.random.randint(0, n, 2)
    while poz[0] == poz[1]:
        poz = numpy.random.randint(0, n, 2)
    p1 = numpy.min(poz)
    p2 = numpy.max(poz)
    y = x.copy()
    y[p1] = x[p2]
    y[p2] = x[p1]
    return y


def mutatie_populatie(pop, dim, n, probabilitate_m):
    pop_m = pop.copy()
    for i in range(dim):
        r = numpy.random.uniform(0, 1)
        if r <= probabilitate_m:
            x = pop[i][0:n].copy()
            x = m_perm_interschimbare(x, n)
            fitness = ok(x, conflicte)
            x = list(x)
            x = x + [fitness]
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


def GA(dim, NMAX, pc, pm):
    pop_initiala = gen(dim, conflicte)
    pop_initiala = numpy.asarray(pop_initiala)
    istoric_v = [numpy.max(pop_initiala[:, -1])]
    it = 0
    gata = False
    nrm = 1
    n = 10
    while it < NMAX:

        parinti = ruleta(pop_initiala, dim, n)
        pop_copii = crossover_populatie(parinti, dim, n, pc)
        pop_copii_mutanti = mutatie_populatie(pop_copii, dim, n, pm)
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

        # salvez cel mai bun individ in istoric
        istoric_v.append(numpy.max(pop_urmatoare[:, -1]))

        pop_initiala = pop_urmatoare.copy()

    poz_max = numpy.where(pop_urmatoare[:, -1] == maxim)
    individ_max_gene = pop_urmatoare[poz_max[0][0], 0:n]
    individ_max_fitness = maxim
    return numpy.asarray(individ_max_gene), individ_max_fitness


if __name__ == "__main__":
    individ_max_gene, individ_max_fitness = GA(10, 10, 0.8, 0.1)
    print(numpy.asarray(individ_max_gene))
    print(individ_max_fitness)
