import numpy as np
import matplotlib.pyplot as plt

# Load the data from the CSV file
donnees = np.genfromtxt('GEL242-DonneesGenieUdeS.csv', delimiter=',', skip_header=1, dtype=str)

# Separate the data into columns
ids = donnees[:, 0].astype(int)  # Assuming IDs are in the first column
genres = donnees[:, 1]  # Assuming genres are in the second column
cycles = donnees[:, 2].astype(int)  # Assuming cycles are in the third column
grandeurs = donnees[:, 3].astype(float)  # Assuming grandeur is in the fourth column

# Count occurrences of each genre
occurrences = {}
for genre in genres:
    if genre in occurrences:
        occurrences[genre] += 1
    else:
        occurrences[genre] = 1

for cycle in cycles:
    if cycle in occurrences:
        occurrences[cycle] += 1
    else:
        occurrences[cycle] = 1


# Print occurrences of each genre

print("Occurrences de chaque genre selon le cycle:")
print(occurrences)
"""
for genre, count in occurrences.items():
    print(f"Genre: {genre}, Nombre d'occurrences: {count}")
"""
def pourcentBac ():
    total_personnes = sum(occurrences.values())
    pourcent = {}  # Utilisez un nom de variable différent pour le dictionnaire
    for cycle, count in occurrences.items():
        pourcentage = (count / total_personnes) * 100  # Stockez le pourcentage calculé dans une variable différente
        pourcent[cycle] = pourcentage  # Utilisez le dictionnaire pour stocker les pourcentages
        print(f"Cycle: {cycle}, Pourcentage de personne: {pourcentage},%")
    print(pourcent)
    return pourcent

pourcentBac()
#for cycle, count in occurrences.items():
    #print(f"Cycle: {cycle}, Nombre d'occurrences: {count}")


def filles_doc():
    total_filles = occurrences.get('F',0)
    count_filles_doc = 0
    for i in range (len(cycles)):
        if cycles[i] == 3 and genres[i] == 'F':
            count_filles_doc += 1
    pourcentage_filles_doc = (count_filles_doc / total_filles) * 100
    print("Il y a", pourcentage_filles_doc, "% de filles étudiant au Doctorat")
    return pourcentage_filles_doc
filles_doc()
# Analyze the data with NumPy
#diagramme à bande
def digramme1():
    # Données
    cycles_uniques = np.unique(cycles)
    moyennes_grandeur = []
    ecarts_types = []

    # Calcul des moyennes et des écarts-types pour chaque cycle
    for cycle in cycles_uniques:
        grandeurs_cycle = grandeurs[cycles == cycle]
        moyenne_cycle = np.mean(grandeurs_cycle)
        ecart_type_cycle = np.std(grandeurs_cycle)
        moyennes_grandeur.append(moyenne_cycle)
        ecarts_types.append(ecart_type_cycle)
    plt.figure(figsize=(10, 6))
    plt.bar(cycles_uniques, moyennes_grandeur, yerr=ecarts_types, capsize=5, color='skyblue')
    plt.title('Grandeur moyenne des étudiants par cycle d\'étude')
    plt.xlabel('Cycle')
    plt.ylabel('Grandeur moyenne')
    plt.xticks(cycles_uniques)
    plt.grid(True)
    plt.show()
digramme1()

def diagramme2():
    # Données
    cycles_uniques = np.unique(cycles)
    moyennes_grandeur = []
    ecarts_types = []

    # Calcul des moyennes et des écarts-types pour chaque cycle
    for cycle in cycles_uniques:
        grandeurs_cycle = grandeurs[cycles == cycle]
        moyenne_cycle = np.mean(grandeurs_cycle)
        ecart_type_cycle = np.std(grandeurs_cycle)
        moyennes_grandeur.append(moyenne_cycle)
        ecarts_types.append(ecart_type_cycle)
    plt.figure(figsize=(10, 6))
    plt.hist(grandeurs, bins=30, color='skyblue', edgecolor='black')
    plt.title('Distribution des grandeurs des étudiants')
    plt.xlabel('Grandeur')
    plt.ylabel('Fréquence')
    plt.grid(True)
    plt.show()
diagramme2()
def diagramme3():
    nb_etudiants_cycle = {}
    for cycle in cycles:
        if cycle in nb_etudiants_cycle:
            nb_etudiants_cycle[cycle] += 1
        else:
            nb_etudiants_cycle[cycle] = 1

    # Création du diagramme à camembert
    plt.figure(figsize=(8, 8))
    plt.pie(nb_etudiants_cycle.values(), labels=nb_etudiants_cycle.keys(), autopct='%1.1f%%', startangle=140)
    plt.title('Répartition des étudiants par cycle d\'études')
    plt.xlabel('Cycle')
    plt.legend(loc='best')
    plt.show()
diagramme3()

"""
moyenne_grandeur = np.mean(grandeurs)
ecart_type_grandeur = np.std(grandeurs)
max_grandeur = np.max(grandeurs)
min_grandeur = np.min(grandeurs)

# Print the results
print("\nMoyenne de la grandeur :", moyenne_grandeur)
print("Écart-type de la grandeur :", ecart_type_grandeur)
print("Maximum de la grandeur :", max_grandeur)
print("Minimum de la grandeur :", min_grandeur)

# Visualize the data with Matplotlib
plt.hist(grandeurs, bins=100, color='skyblue', edgecolor='black')
plt.title('Distribution des grandeurs des deux genres')
plt.xlabel('Grandeur')
plt.ylabel('Fréquence')
plt.xlim(160,190)
plt.grid(True)
plt.show()


for genre in np.unique(genres):
    # Filter data for the current genre
    mask = genres == genre
    grandeurs_genre = grandeurs[mask]

    # Calculate statistics for the current genre
    moyenne_grandeur = np.mean(grandeurs_genre)
    ecart_type_grandeur = np.std(grandeurs_genre)
    max_grandeur = np.max(grandeurs_genre)
    min_grandeur = np.min(grandeurs_genre)

    # Print the results
    print(f"\nGenre: {genre}")
    print("Moyenne de la grandeur :", moyenne_grandeur)
    print("Écart-type de la grandeur :", ecart_type_grandeur)
    print("Maximum de la grandeur :", max_grandeur)
    print("Minimum de la grandeur :", min_grandeur)

    # Visualize the data with Matplotlib
    plt.hist(grandeurs_genre, bins=10, color='skyblue', edgecolor='black')
    plt.title(f'Distribution des grandeurs pour le genre {genre}')
    plt.xlabel('Grandeur')
    plt.ylabel('Fréquence')
    plt.grid(True)
    plt.show()
# Initialize variables to count occurrences of large gaps between consecutive girls
large_gap_count = 0

# Iterate over the grandeurs and check for large gaps between consecutive girls
for i in range(len(grandeurs) - 1):
    if cycles[i] == 3 and cycles[i+1] == 3:
        if genres[i] == 'F' and genres[i+1] == 'F':  # Check if both are girls
            gap = abs(grandeurs[i] - grandeurs[i+1])
            if gap > 2.5:  # Check if the gap is greater than 10 cm
                large_gap_count += 1

# Print the total number of large gaps between consecutive girls
print("Nombre d'écarts de plus de 10 cm entre deux filles consécutives :", large_gap_count)
def average_grandeur_by_genre(donnees):

    #Calculate the average grandeur based on the genre of the person.

    #Args:
    #- donnees: NumPy array containing the data, with genres as the second column and grandeur as the fourth column.

    #Returns:
    #- average_grandeur_H: Average grandeur for males
    #- average_grandeur_F: Average grandeur for females

    # Separate the data into columns
    genres = donnees[:, 1]  # Assuming genres are in the second column
    grandeurs = donnees[:, 3].astype(float)  # Assuming grandeur is in the fourth column

    # Filter grandeur based on genres
    grandeurs_H = grandeurs[genres == 'H']
    grandeurs_F = grandeurs[genres == 'F']

    # Calculate average grandeur for each genre if data is available
    average_grandeur_H = np.mean(grandeurs_H) if grandeurs_H.size > 0 else np.nan
    average_grandeur_F = np.mean(grandeurs_F) if grandeurs_F.size > 0 else np.nan

    return average_grandeur_H, average_grandeur_F

# Example usage
average_grandeur_H, average_grandeur_F = average_grandeur_by_genre(donnees)
print("\nAverage grandeur for males:", average_grandeur_H)
print("Average grandeur for females:", average_grandeur_F)"""
