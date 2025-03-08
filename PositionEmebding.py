import torch
import math


class PositionEmbedding(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
    
    # pos est la position du token dans la phrase
    # i est la dimension de l'embedding
    # d_model est la taille de l'embedding
        def get_pe(pos,i,d_model):
            fenmu = 1e4**(i/d_model)
            pe = pos/fenmu

            if i%2 == 0:
                return math.sin(pe)
            return math.cos(pe)
    

    # on utilise empty pour ne pas initialiser les valeurs
    # on va remplir les valeurs de pe avec la fonction get_pe
    # pe est de taille 50 x 32
    # càd 50 tokens avec un embedding de taille 32
    # ici 50 est la taille maximale du nombre de tokens dans une phrase
        pe = torch.empty(50,32)
        for i in range(50):
            for j in range(32):
                pe[i,j] = get_pe(i,j,32)
        pe = pe.unsqueeze(0)

    # pe.unsqueeze(0) pour ajouter une dimension au début
    # pour ensuite replacer la batch_size à la première position


        self.register_buffer('pe', pe)
    # Enregistre 'pe' comme un buffer qui ne participe pas à la mise à jour 
    # du gradient. Cela permet de le sauvegarder et de le déplacer (par exemple sur GPU) 
    # avec le modèle, sans l'entraîner.

        self.embed = torch.nn.Embedding(39, 32)
    # Couche d'embedding (représentation vectorielle des tokens)
    # 39 indique la taille du vocabulaire (nombre total de tokens distincts, indices 0 à 38).
    # 32 indique la dimension de l'embedding (chaque token est représenté par un vecteur de taille 32)


        self.embed.weight.data.normal_(0, 0.1)
    # Initialisation des poids de la couche d'embedding
    # On applique une distribution normale centrée à 0 avec un écart-type (std) de 0.1.

    def forward(self, x):
        embed = self.embed(x)
        # [8, 50] -> [8, 50, 32]
        # Convertir les indices de tokens (8, 50) en vecteurs d'embedding (8, 50, 32)

        embed = embed + self.pe
        # Ajouter les vecteurs de position aux vecteurs d'embedding
        # afin d'injecter la notion de position dans la représentation de chaque token
        return embed