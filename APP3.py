import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# Load the data from the CSV file
donnees_1 = np.genfromtxt('S2GE_APP3_Problematique_Detecteur_Primaire.csv', delimiter=',', skip_header=1, dtype=str)

# Separate the data into columns
index_1 = donnees_1[:, 0].astype(int)  # Assuming IDs are in the first column
temps_1 = donnees_1[:, 1].astype(float)  # Assuming genres are in the second column
tension_1 = donnees_1[:, 2].astype(float)  # Assuming genres are in the second column
temps_mort_1 = donnees_1[:, 3].astype(int)  # Assuming cycles are in the third column
temperature_1 = donnees_1[:, 4].astype(int)  # Assuming grandeur is in the fourth column

# Load the data from the CSV file
donnees_2 = np.genfromtxt('S2GE_APP3_Problematique_Detecteur_Secondaire.csv', delimiter=',', skip_header=1, dtype=str)

# Separate the data into columns
index_2 = donnees_2[:, 0].astype(int)  # Assuming IDs are in the first column
temps_2 = donnees_2[:, 1].astype(float)  # Assuming genres are in the second column
tension_2 = donnees_2[:, 2].astype(float)  # Assuming genres are in the second column
temps_mort_2 = donnees_2[:, 3].astype(int)  # Assuming cycles are in the third column
temperature_2 = donnees_2[:, 4].astype(int)  # Assuming grandeur is in the fourth column

somme_temps_total_1 = np.sum(temps_1)
print(f"Voici le temps total{somme_temps_total_1}")
# Calculer les différences entre les éléments consécutifs de temps_1
somme_temps_total_2 = np.sum(temps_2)
print(f"Voici le temps total{somme_temps_total_2}")
differences = np.diff(temps_1)
total_temps_mort = np.sum(temps_mort_1)
print(total_temps_mort)
# Trouver la plus petite différence
plus_petite_difference = np.min(differences)


print("Plus petite différence entre les valeurs consécutives de temps_1 :", plus_petite_difference)

# Calculer les différences entre les éléments consécutifs de temps_1
differences = np.diff(temps_2)
Valeur_importante = 1/(46920574 - total_temps_mort)*len(index_1)
print(f"La valeur importante est {Valeur_importante}")
# Trouver la plus petite différence
plus_petite_difference = np.min(differences)

print("Plus petite différence entre les valeurs consécutives de temps_2 :", plus_petite_difference)

highest_value = np.max(temps_1)
print("Highest value in temps_1:", highest_value)



def calculate_coincidence(temps_1, temps_2, index_1):
    Tau = 0.01
    nB_val = 0
    coincidence = np.zeros(len(temps_1))

    for i in range(len(index_1)):
        j = i - 165
        while coincidence[i] == 0 and j <= 38166:
            if np.abs(temps_2[j] - temps_1[i]) < Tau:
                coincidence[i] = 1
                nB_val += 1
                break
            if j >= (i + 165) or j >= 38166:
                break
            j += 1

    coincident_indices = np.where(coincidence == 1)[0]
    coincident = tension_1[coincident_indices]
    noncoincident = tension_1[np.where(coincidence == 0)[0]]
    print(nB_val)
    return coincident, noncoincident


coincident, noncoincident = calculate_coincidence(temps_1, temps_2, index_1)

def calculate_coincidence_avec_temps_mort(temps_1, temps_2, index_1):
    temps_mort = 0.01
    nB_val = 0
    coincidence = np.zeros(len(temps_1))

    for i in range(len(index_1)):
        j = i - 165
        while coincidence[i] == 0 and j <= 38166:
            if np.abs(temps_2[j] - temps_1[i]) < temps_mort:
                coincidence[i] = 1
                nB_val += 1
                break
            if j >= (i + 165) or j >= 38166:
                break
            j += 1

    coincident_indices = np.where(coincidence == 1)[0]
    coincident = tension_1[coincident_indices]
    noncoincident = tension_1[np.where(coincidence == 0)[0]]

    return coincident, noncoincident
calculate_coincidence_avec_temps_mort(temps_1, temps_2,index_1)

coincident_corriger, noncoincident_corrige = calculate_coincidence_avec_temps_mort(temps_1, temps_2, index_1)


def plot_histogram(tension_1, tension_2, coincident, noncoincident):
    # Example data
    temps = max(temps_1) * 1 ** -3
    facteur1 = [1 / temps] * (len(tension_1))
    plt.figure(figsize=(8, 6), facecolor='w')

    facteur2 = [1 / temps] * (len(coincident))
    facteur3 = [1 / temps] * (len(noncoincident))

    # Calculate mean and standard deviation for coincident values
    mean_coincident = np.mean(coincident)
    std_coincident = np.std(coincident, ddof=1)  # Corrected to use unbiased estimator (N-1)

    # Define bins
    bins = np.logspace(np.log10(tension_1.min()), np.log10(tension_1.max()), 25)
    colors = ['#FF5733', '#ccebc5', '#a8ddb5', '#7bccc4', '#CCCCCC']

    # Plot histogram for tension_2
    plt.hist(tension_1, bins=bins, histtype='step', color='cornflowerblue', linewidth = 2,weights = facteur1)

    # Plot histogram for non-coincident values
    plt.hist(noncoincident, bins=bins, color='orchid', histtype='step', linestyle='-', linewidth=2,weights = facteur3)

    # Plot histogram for coincident values
    plt.hist(coincident, bins=bins, color=colors[2], histtype='step', linestyle='-', linewidth=1,weights = facteur2)
    #hist, bins = np.histogram(coincident, bins=bins)
    # Calculate the error bars
    #error = np.square(hist)

    #bin_centers = (bins[:-1] + bins[1:]) / 2
    #bin_width = bins[1] - bins[0]
    #yerr = np.sqrt(std_coincident) / bin_width

    hist, bins = np.histogram(coincident, bins=bins)
    error = np.sqrt(hist)/(46920574)

    # Calculate bin centers and width
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]

    # Plot error bars for coincident values without dots
    plt.errorbar(bin_centers, hist*(1/46920574), yerr=error, fmt='', color=colors[2], linestyle= '', capsize=9.5, capthick=4, linewidth = 4 )
   # plt.vlines(bin_centers, hist - yerr, hist + yerr, colors='r', linewidth=1)
    plt.xscale('log')
    plt.xticks([10, 100], ['10¹', '10²'])

    plt.title("Histogramme de la fréquence en fonction de la tension")
    plt.xlabel("Tension [mV]")
    plt.ylabel("Fréquence/bin[s⁻¹]")
    plt.grid(alpha = 0.15)
    #plt.gcf().set_size_inches(10, 5)
    # Add legend
    custom_lines = [
        Line2D([0], [0], color='cornflowerblue', linestyle='-', linewidth=2),
        Line2D([0], [0], color='orchid', linestyle='-', linewidth=2),
        Line2D([0], [0], color=colors[2], linestyle='-', linewidth=2),

    ]
    plt.legend(custom_lines, ['All events', 'Non-coincident', 'Coincident'], loc='best',
               frameon=False)
    plt.box()
    #plt.ylim(0, 5500)
    plt.xlim(right = 500+10**2)
    plt.show()
    plt.savefig('Beau.png', dpi=300)

# Example usage:
plot_histogram(tension_1, tension_2, coincident, noncoincident)

def plot_histogram_corriger(tension_1, tension_2, coincident_corriger, noncoincident_corrige):
    bins = np.logspace(np.log10(tension_1.min()), np.log10(tension_1.max()), 25)
    temps = max(temps_1)*1**-3
    #total_weight = 1 / 46920574.589
    #weight_factor = total_weight / len(tension_1)

    temps_mort_total = 0
    for k in range(temps_1.shape[0]):
        temps_mort_total += temps_mort_1[k]
    # Plot histogram for tension_2
    facteur1 = [1 /(temps-temps_mort_total)] * (len(tension_1))

    plt.hist(tension_1, bins=bins, edgecolor='g', histtype='step', weights= facteur1)
    facteur2 = [1 / (temps-temps_mort_total)] * (len(noncoincident_corrige))
    facteur3 = [1 / (temps-temps_mort_total)] * (len(coincident_corriger))

    # Plot histogram for non-coincident values
    plt.hist(noncoincident_corrige, bins=bins, color='gray', histtype='step', linestyle='--', linewidth=2, weights= facteur2)

    # Calculate mean and standard deviation for coincident values
    mean_coincident = np.mean(coincident_corriger)
    std_coincident = np.std(coincident_corriger, ddof=1)  # Corrected to use unbiased estimator (N-1)

    # Plot histogram for coincident values
    plt.hist(coincident_corriger, bins=bins, color='r', histtype='step', linestyle='--', linewidth=2, weights= facteur3)

    # Calculate the error bars
    hist, bins = np.histogram(coincident_corriger, bins=bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]
    yerr = np.sqrt(hist) / bin_width

    # Plot error bars for coincident values with more visibility
    #plt.errorbar(bin_centers, hist, yerr=yerr, fmt='o', color='r', markersize=1, capsize=1, capthick=1, linewidth=1)

    plt.xscale('log')
    plt.xticks([10, 100], ['10¹', '10²'])
    plt.title("Histogram of Amplitude - Corrected")
    plt.xlabel("calculated SiPM peak voltage [mV]")
    plt.ylabel("Rate/bin[s⁻¹]")
    custom_lines = [
        Line2D([0], [0], color='g', linestyle='-', linewidth=2),
        Line2D([0], [0], color='gray', linestyle='--', linewidth=2),
        Line2D([0], [0], color='r', linestyle='--', linewidth=2),
    ]
    plt.legend(custom_lines, ['All events', 'Non-coincident', 'Coincident corrigé'])
    plt.grid(True)

    plt.show()

plot_histogram_corriger(tension_1, tension_2, coincident_corriger, noncoincident_corrige)
def estimate_blind_time(temps_1, temps_2):
    """
    Estimate the blind time of the system based on the data.

    Parameters:
    - temps_1 (numpy.ndarray): Array containing timestamps for primary detector.
    - temps_2 (numpy.ndarray): Array containing timestamps for secondary detector.

    Returns:
    - blind_time (float): Estimated blind time of the system.
    """
    # Find the gaps between consecutive timestamps for both detectors
    gaps_1 = np.diff(temps_1)
    gaps_2 = np.diff(temps_2)

    # Estimate the blind time as the maximum gap between consecutive timestamps
    blind_time = max(np.max(gaps_1), np.max(gaps_2))
    return blind_time

blind_time = estimate_blind_time(temps_1, temps_2)
print("Estimated blind time:", blind_time)
estimate_blind_time(temps_1, temps_2)
