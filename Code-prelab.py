import numpy as np
import matplotlib as plt
N = 4
LIGNES = 6
#fonction de test unitaire
def ValideResultat(reponseTrouvee, reponseAttendue):
    for k in range (0, N):
        if (reponseTrouvee[k] != reponseAttendue[k]): 
            return False
    
    return True
reponseTrouvee = [1, 2, 3, 4]
reponseAttendue = [1, 2, 3, 4]
est_valide = ValideResultat(reponseTrouvee,reponseAttendue)
print(est_valide)

#fonction de calcul
def CalculBoucle(metrique, etat_initial, etat_final):
    for k in range (0,N):
        etat_final[k] = 250
        for u in range (0, N):
            temp = metrique[k*N+u] + etat_initial[u]
            if (temp < etat_final[k]):
                etat_final[k] = temp

if __name__ == "__main__":
    ref_metriques = np.array([
        [4, 3, 3, 2, 0, 3, 5, 4, 4, 3, 3, 2, 2, 5, 3, 2],
        [3, 4, 2, 3, 5, 2, 2, 3, 3, 4, 2, 3, 3, 0, 4, 5],
        [4, 5, 3, 0, 2, 3, 3, 4, 2, 3, 5, 2, 2, 3, 3, 4],
        [2, 5, 3, 2, 4, 3, 3, 2, 0, 3, 5, 4, 4, 3, 3, 2],
        [3, 2, 4, 3, 5, 4, 0, 3, 3, 2, 4, 3, 3, 2, 2, 5],
        [3, 4, 2, 3, 3, 0, 4, 5, 3, 4, 2, 3, 5, 2, 2, 3]
    ])

    resultats_attendus = np.array([
        [2, 0, 2, 2],
        [4, 2, 4, 0],
        [0, 4, 2, 4],
        [2, 4, 0, 4],
        [4, 0, 4, 2],
        [4, 0, 4, 2]
    ])

    erreurTrouvee = False
    etat_actuel = np.zeros(N, dtype=int)
    etat_nouveau = np.zeros(N, dtype=int)
    for k in range (0, LIGNES):
        CalculBoucle(ref_metriques[k], etat_actuel, etat_nouveau)
        print(etat_nouveau)

        if (False == ValideResultat(etat_nouveau, resultats_attendus[k])):
            erreurTrouvee = True
        
        etat_actuel = etat_nouveau.copy()
    
    if (erreurTrouvee):
        print("Il y a une erreur de calcul")
    else:
        print("Le calcul s'excécute correctement")2 0 2 2 
4 2 4 0 
0 4 2 4 
2 4 0 4 
4 0 4 2 
4 0 4 2 
Le calcul s'exécute correctement.

    

        
