import numpy as np

# Datele problemei
N = 10  # Exemplu de N, înlocuiți cu valoarea reală
CONFLICT = np.random.randint(0, 2, (N, N))
np.fill_diagonal(CONFLICT, 0)

########################## FUNCTIE FITNESS + FEZABILITATE ########################
def calculeaza_conflicte(permutare):
    return np.sum([CONFLICT[permutare[i], permutare[(i+1) % N]] for i in range(N)])

def fitness(permutare):
    return 1 / (1 + calculeaza_conflicte(permutare))

########################## GENERAREA UNEI POPULATII ########################
def gen_populatie(dim):
    pop = []
    for _ in range(dim):
        permutare = np.random.permutation(N)
        pop.append(np.append(permutare, fitness(permutare)))
    return np.array(pop)

pop = gen_populatie(100)
print(pop)