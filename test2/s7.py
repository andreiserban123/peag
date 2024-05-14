import numpy as np

# Datele problemei
cantitati_disponibile = np.array([100, 80, 120, 50])  # smochine, ananas, curmale, merișor
proportii_combinatii = np.array([
    [0.3, 0.25, 0.25, 0.2],   # Combinația 1
    [0.25, 0, 0.75, 0],       # Combinația 2
    [0.25, 0.25, 0.25, 0.25], # Combinația 3
    [0, 0, 1, 0],             # Combinația 4
    [1, 0, 0, 0]              # Combinația 5
])
profituri = np.array([20, 10, 15, 12, 5])

########################## FUNCTIE FITNESS + FEZABILITATE ########################
def calculeaza_profit(X):
    return np.sum(X * profituri)

def is_feasible(X):
    utilizare = np.sum(proportii_combinatii * X[:, np.newaxis] / 5, axis=0)  # Transformăm gramele în kilograme
    return np.all(utilizare <= cantitati_disponibile)

def fitness(X):
    if not is_feasible(X):
        return -float('inf')
    return calculeaza_profit(X)

########################## GENERAREA UNEI POPULATII ########################
def gen_populatie(dim):
    pop = []
    while len(pop) < dim:
        X = np.random.randint(0, 100, 5)  # Generăm cantități între 0 și 100
        if is_feasible(X):
            pop.append(np.append(X, fitness(X)))
    return np.array(pop)

pop = gen_populatie(100)
print(pop)