
import torch
from Data import zidian_x, zidian_y

def mask_pad(data):
    mask = data == zidian_x['<PAD>']
    # on crée un masque qui vaut True si data est égal à '<PAD>'

    # si on a b phrases de taille 50, on a mask de taille b x 50
    # on veut un masque de taille b x 1 x 1 x 50
    # [b, 50] -> [b, 1, 1, 50]
    mask = mask.reshape(-1,1,1,50)
    # on reshape pour avoir la même taille que data
    # reshape(-1, 1, 1, 50) 
    # -1 pour la batch size
    # 1 pour le nombre de heads d'attention
    # 1 pour la query_length 
    # 50 pour la key_length 


    # Quand on veut calculer l'attention, on calcule l'attention entre 50 mots et 50 mots
    # Donc la taille de la matrice d'attention est 50 x 50
    # la colonne de pad est true, càd on ne veut pas prendre en compte les mots de padding
    # Mais pad lui-même par rapport aux autres mots, l'attention n'est pas nulle
    # Donc la ligne de pad est false


    mask = mask.expand(-1, 1,50,50)
    #
    return mask 

def mask_triu(data):
    tril = 1-torch.tril(torch.ones(1,50,50),dtype = torch.long)
    mask = data == zidian_y['<PAD>']

    mask = mask.unsqueeze(1).long()

    mask = mask + tril

    mask = mask > 0

    mask = (mask == 1).unsqueeze(dim=1)

    return mask
