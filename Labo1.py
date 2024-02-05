import argparse
import random
def guess_number(valeurSecrete,valeurMin,valeurMax):


    #demande à l'utilisateur de deviner la valeur secrète
    tentative = 0
    chance = 10
    while tentative < 10:
        try:
            guess = int(input("Entrez un nombre entre {} et {}: ".format(valeurMin,valeurMax)))
        except ValueError:
            print("Vous n'avez pas entrez un nombre entre {} et {}".format(valeurMin,valeurMax))
            continue
        if guess < valeurMin or guess > valeurMax:
            print("Vous n'avez pas entrez un nombre allant de {} à {}".format(valeurMin, valeurMax))
            continue
        #regarder si le nombre est correct
        if guess > valeurSecrete:
            print("Vous avez entrer une valeur trop grande")
        elif guess < valeurSecrete:
            print("Vous avez entrer une valeur trop petite")
        else:
            print("Félicitation! Tu as réussi!")
            return
        tentative += 1
        chance = chance - 1
        print("il reste ", chance, "tentatives")
    print("Désolé, vous avez dépassé le nombre de tentative autorisé. La valeur secrète était {}".format(valeurSecrete))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Jeu du nombre secret')
    parser.add_argument('--min', type= int, default=1, help="Le minimum du nombre secret est entre 1 et 10")
    parser.add_argument('--max', type= int, default=10, help="Le maximum du nombre secret est entre 1 et 10")
    args = parser.parse_args()

    #choix du nombre secret
    valeurSecrete = random.randint(args.min, args.max)
    guess_number(valeurSecrete,args.min,args.max)
