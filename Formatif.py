import argparse


import numpy as np
import matplotlib.pyplot as plt
import argparse as args
#Question 1 : Charger en mémoire la totalité des échantillons dans un tableau Numpy
donnees = np.genfromtxt('S2GE_APP3_Examen_Formatif_Donnees.csv', delimiter=',', skip_header=1)

index = donnees[:,0].astype(int)
amplitude = donnees[:,1].astype(int)
#Question 2: Créer un vecteur contenant les classes (bins) pour les histogrammes demandés plus loin. Il
#doit avoir 64 classes, où les valeurs sont également réparties entre 0 et 1024.
array_index = np.array(index)
array_amplitude = np.array(amplitude)
print(array_index)
print(index)
print(amplitude)
echantillon = (np.concatenate((array_index, amplitude)))
print(echantillon)
nb_classe = 64
#print(max(amplitude))
#print(min(amplitude))

#Q3: Afficher dans un graphique l’histogramme de l’amplitude avec les classes (bins) prédéfinies
#ci-haut. Vous obtiendrez une forme similaire à la figure 1.
bins = np.linspace(min(amplitude), max(amplitude), nb_classe+1)
#print(bins)
plt.figure()
plt.hist(amplitude, bins=bins)
#Q4 : Donnez un nom aux axes du graphique et ajoutez le titre "Histogramme des amplitudes".
plt.title("Histogramme des amplitudes")
plt.xlabel("")
#Q5:Avec la méthode de votre choix, obtenir sous forme de vecteur Numpy l’histogramme des
#amplitudes, également avec 64 classes également réparties entre 0 et 1023 (inclusivement).
classe_bins, _ = np.histogram(amplitude, bins=bins)
#print(classe_bins)
plt.show()
"""
6. Complétez la fonction TrouverMaxLocal.
(a) Sans utiliser les fonctions Numpy, à l’aide de structures de contrôle standards
(for, if, while, etc.), parcourir le vecteur d’histogramme passé en paramètre par la fin et
identifier programmatiquement la position du premier maximum local trouvé ainsi que
sa valeur.
(b) Retourner la valeur du max et sa position dans l’histogramme
"""

def TrouverMaxLocal(classes_bins):
    index_position = 0
    val_max = classes_bins[0]
    for k in range(1, len(classes_bins)):
        if val_max < classes_bins[k]:
            index_position = k
            val_max = classes_bins[k]
    print(f"La valeur du max est de: {val_max} à l'index : {index_position}")
    return val_max
TrouverMaxLocal(classe_bins)
val_max = TrouverMaxLocal(classe_bins)
#Q8: Sur l’histogramme d’amplitudes, ajoutez une barre horizontal à la hauteur trouvée. La
#barre horizontale doit, en largeur, aller de 0 jusqu’à la valeur maximale des classes de
#l’histogramme.
classe_bins, _ = np.histogram(amplitude, bins=bins)

plt.hist(amplitude, bins=bins)
plt.hlines(2600, 0,47)
#print(classe_bins)
plt.show()

parser = argparse.ArgumentParser(description="Formatif cancer")
parser.add_argument("--ecrasetitre",type = str ,default=None, help="Tu fais quoi bro?")
args = parser.parse_args()
titre = args.ecrasetitre if args.ecrasetitre is not None else "Histogramme des amplitudes"

# Afficher l'histogramme avec le titre déterminé
plt.figure()
plt.hist(amplitude, bins=bins)
plt.title(titre)
plt.xlabel("Amplitude")
plt.ylabel("Fréquence")
plt.savefig("MALG1102_histoAmplitude.png")
plt.show()  # Afficher le graphique une seule fois

