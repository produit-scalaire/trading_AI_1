
import numpy as np
import pickle

# Ouvrir le fichier en mode lecture binaire
with open("dictionnaire/dict_data.pickle", "rb") as fichier:
    dict_data = pickle.load(fichier)

# Maintenant, mon_dictionnaire contient les données chargées

###on crée le reseaux de neurones
#on definit la dimension des couches de neuroens

import torch
import torch.nn as nn
import reseaux_neurones

input_size = 256
hidden_size1 = 128
hidden_size2 = 64
output_size = 32

device = "cuda" if torch.cuda.is_available() else "cpu"
model = reseaux_neurones.neural_net_stock_market(input_size, hidden_size1, hidden_size2, output_size).to(device)


###Trian model
#fonction loss
loss_MSE = nn.MSELoss()

#on cree l'optimizer
optimizer = torch.optim.Adam(params = model.parameters(), lr = 0.01)

###on cree la boucle
def Train():
    num_epoch = 1
    i = 0
    for epoch in range (num_epoch):
        for stock in dict_data:
            v = 256

            #print(dict_data[stock])
            #print(len(dict_data[stock]))
            #print(stock)
            X_train = []
            pas_assez_de_données = False
            for index in range(v):
                n = index
                while len(dict_data[stock][n][2]) == 0:
                    if n + 1 == len(dict_data[stock]) or n == v + 10:
                        pas_assez_de_données = True
                        break
                    else:
                        n += 1
                if dict_data[stock][index][2] == '':
                    break
                X_train.append(float(dict_data[stock][index][2]))
            while (v + 33) < len(dict_data[stock]) or not pas_assez_de_données:
                model.train()
                #1
                #print(dict_data[stock][v + 32])
                X_train = torch.tensor(X_train, dtype=torch.float32)
                y_pred = model(X_train)
                y_train = []
                n = 0
                if len(dict_data[stock]) <= (v + 32):
                    break
                for index in range (v + 1, v + 33):
                    n = index
                    #print(len(dict_data[stock])-v)
                    while len(dict_data[stock][n][2]) == 0:
                        n -= 1
                    y_train.append(float(dict_data[stock][n][2]))
                y_train = torch.tensor(y_train, dtype=torch.float32)

                #2
                loss = loss_MSE(y_pred, y_train)

                #3
                optimizer.zero_grad()

                #4
                loss.backward()

                #5
                optimizer.step()

                X_train = list(X_train)
                X_train.pop(0)
                n = v
                n_est_trop_grand = False
                while len(dict_data[stock][n][2]) == 0:
                    if n + 1 == len(dict_data[stock]) or n == v + 10:
                        n_est_trop_grand = True
                        break
                    else:
                        n += 1
                if n_est_trop_grand:
                    break
                X_train.append(float(dict_data[stock][n][2]))

                i += 1
                v += 1
                if i % 1 == 0:
                    print(f"iterations: {i} | MSE Train Loss: {loss}")
    print(f"l'entrainement est terminé nombre d'iterations = {i} nombre d'epochs = {num_epoch}")

Train()

from pathlib import Path

MODEL_PATH = "models"

# 2. Create model save path
MODEL_NAME = "model_AI_trading.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 3. Save the model state dict
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model.state_dict(), # only saving the state_dict() only saves the models learned parameters
           f=MODEL_SAVE_PATH)


