import pickle
import csv
import os

###on ouvre les données et on crée un dictionnaire avec

# Chemin vers le répertoire contenant les fichiers CSV
repertoire = 'stock_market_data/nasdaq/csv'

#dictionaire de tout les dossiers
dict_data = {}


# Parcours de tous les fichiers dans le répertoire
for nom_fichier in os.listdir(repertoire):
    if nom_fichier.endswith('.csv'):
        dict_data[nom_fichier[:-4]] = []
        chemin_fichier = os.path.join(repertoire, nom_fichier)
        f = open(chemin_fichier, 'r')
        les_lignes = f.readlines()
        f.close()

        for ligne in les_lignes:
            ligne = ligne.strip()
            dict_data[nom_fichier[:-4]].append(ligne.split(','))
        if  len(dict_data[nom_fichier[:-4]]) < 365*2 :
            dict_data.pop(nom_fichier[:-4], None)
        else:
            dict_data[nom_fichier[:-4]].pop(0)

print("le dictionnaire est calculé")

#sauvegarde le dossier

#ouvrir le fichier en mode lecture binaire
with open("dictionnaire/dict_data.pickle", "wb") as fichier:
    pickle.dump(dict_data, fichier)
