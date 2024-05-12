import numpy as np


def ok(x, n, cmax, c, p, o):
    sum_cursuri = np.sum(x)
    if sum_cursuri == 0:
        return False, 0  # Avoid division by zero
    cost = np.dot(c, x)
    puncte = np.dot(p, x)
    ore = np.dot(o, x)

    puncte_medii = puncte / sum_cursuri
    ore_medii = ore / sum_cursuri
    if ore_medii > 70 or cost > cmax:
        return False, puncte_medii
    return True, puncte_medii


def gen(dim, cmax):
    c = np.array([1000, 800, 1500], dtype=int)
    p = np.array([5, 3, 8], dtype=int)
    o = np.array([80, 40, 100], dtype=int)
    n = len(c)
    pop = []
    for i in range(dim):
        gata = False
        while not gata:
            x = np.random.randint(0, 11, n)
            gata, fitness = ok(x, n, cmax, c, p, o)
        x = list(x)
        x.append(fitness)
        pop.append(x)
    return pop


pop = gen(100, 20000)
pop = np.asarray(pop)
print(pop)
