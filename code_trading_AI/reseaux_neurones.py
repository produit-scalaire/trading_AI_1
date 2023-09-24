import torch
import torch.nn as nn

#reseau de neurones pour le chess bot avec apprentissage guidé

class chess_neural_network(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(chess_neural_network, self).__init__()

        # Initialisation des couches
        self.layers = nn.ModuleDict()
        self.layers["layer_1"] = nn.Linear(input_size, hidden_size1)
        self.layers["layer_2"] = nn.Linear(hidden_size1, hidden_size2)
        for i in range(3, 81):
            self.layers[f"layer_{i}"] = nn.Linear(hidden_size2, hidden_size2)
        self.layers["layer_80"] = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        for layer_name in self.layers:
            layer = self.layers[layer_name]
            x = layer(x)
            x = torch.relu(x)
        return x

# Création du réseau de neurones pour l'apprentissage par renforcement du jeu d'echecs
class reseau_neurones(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(reseau_neurones, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 256)
        self.fc3 = nn.Linear(256, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = torch.relu(x)
        return x

# Création du réseau de neurones pour la prediction de la bourse

class neural_net_stock_market(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(neural_net_stock_market, self).__init__()
        # Initialisation des couches
        self.layers = nn.ModuleDict()
        self.layers["layer_1"] = nn.Linear(input_size, hidden_size1)
        self.layers["layer_2"] = nn.Linear(hidden_size1, hidden_size2)
        for i in range(3, 30):
            self.layers[f"layer_{i}"] = nn.Linear(hidden_size2, hidden_size2)
        self.layers["layer_30"] = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        for layer_name in self.layers:
            layer = self.layers[layer_name]
            x = layer(x)
            x = torch.relu(x)
        return x
