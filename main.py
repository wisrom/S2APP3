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

# Print occurrences of each genre
print("Occurrences de chaque genre:")
for genre, count in occurrences.items():
    print(f"Genre: {genre}, Nombre d'occurrences: {count}")

# Analyze the data with NumPy
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
plt.hist(grandeurs, bins=10, color='skyblue', edgecolor='black')
plt.title('Distribution des grandeurs des deux genres')
plt.xlabel('Grandeur')
plt.ylabel('Fréquence')
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

def average_grandeur_by_genre(donnees):
    """
    Calculate the average grandeur based on the genre of the person.

    Args:
    - donnees: NumPy array containing the data, with genres as the second column and grandeur as the fourth column.

    Returns:
    - average_grandeur_H: Average grandeur for males
    - average_grandeur_F: Average grandeur for females
    """
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
print("Average grandeur for females:", average_grandeur_F)
