import numpy as np
import matplotlib.pyplot as plt


# Load the data from the CSV file
donnees_1 = np.genfromtxt('S2GE_APP3_Problematique_Detecteur_Primaire.csv', delimiter=',', skip_header=1, dtype=str)

# Separate the data into columns
index_1 = donnees_1[:, 0].astype(int)  # Assuming IDs are in the first column
temps_1 = donnees_1[:, 1].astype(float)  # Assuming genres are in the second column
tension_1 = donnees_1[:, 2].astype(float)  # Assuming genres are in the second column
temps_mort_1 = donnees_1[:, 3].astype(int)  # Assuming cycles are in the third column
temperature_1 = donnees_1[:, 4].astype(int)  # Assuming grandeur is in the fourth column

print(temps_1)

# Load the data from the CSV file
donnees_2 = np.genfromtxt('S2GE_APP3_Problematique_Detecteur_Secondaire.csv', delimiter=',', skip_header=1, dtype=str)

# Separate the data into columns
index_2 = donnees_2[:, 0].astype(int)  # Assuming IDs are in the first column
temps_2 = donnees_2[:, 1].astype(float)  # Assuming genres are in the second column
tension_2 = donnees_2[:, 2].astype(float)  # Assuming genres are in the second column
temps_mort_2 = donnees_2[:, 3].astype(int)  # Assuming cycles are in the third column
temperature_2 = donnees_2[:, 4].astype(int)  # Assuming grandeur is in the fourth column


def coincidence(temps_1, temps_2):
    tau = 10
    array = np.zeros(len(temps_1))
    nombre = 0
    for i in range(len(temps_1)):
        indices = np.abs(temps_1 - temps_2[i])
        indice_egal = np.where(indices <= tau)
        if np.any(indices <= tau):
            array[i] = temps_1[i]
            print("index", i, temps_1[i], "Temps_2 : ", temps_2[indice_egal])
            nombre = nombre + 1
            print(nombre)
coincidence(temps_1, temps_2)
