import math
import torch


# on définit la fonction d'attention
def attention(Q, K, V, mask):
    """
    Compute 'Scaled Dot Product Attention'
    """

    # on va dire qu'on a b phrases de taille 50, avec un embedding de taille 32
    # et 4 têtes d'attention, donc chaque tête d'attention aura taille 8
    # Q,K,V sont de taille b x 4 x 50 x 8

    # pour calculer l'attention, on va faire un produit scalaire entre Q et K
    # donc on a [b, 4, 50, 50] x [b, 4, 50, 50] -> [b, 4, 50, 50]

    score = torch.matmul(Q, K.permute(0, 1, 3, 2))

    # puis on divise par la racine carrée de la taille de l'embedding
    score = score / math.sqrt(Q.size(-1))  # ici 8**0.5

    # on utilise mask pour ne pas prendre en compte les tokens de padding
    # c'est booléen, on remplace par -inf les valeurs où mask est True
    # alors avec softmax, ces valeurs deviendront 0
    # mask = [b, 1, 50, 50]
    score = score.masked_fill(mask, -float("inf"))
    score = torch.softmax(score, dim=-1)

    # on multiplie par V pour obtenir le résultat final de l'attention
    output = torch.matmul(score, V)

    # on merge les têtes d'attention
    # [b, 4, 50, 8] -> [b, 50, 32]
    output = output.permute(0, 2, 1, 3).reshape(-1, Q.size(2), Q.size(3) * 4)

    return output
