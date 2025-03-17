import math
import torch
from attention import attention

class MultiHead(torch.nn.Module):
    def __int__(self):
        super().__init__()
        self.fc_Q = torch.nn.Linear(in_features=32, out_features=32)
        self.fc_K = torch.nn.Linear(in_features=32, out_features=32)
        self.fc_V = torch.nn.Linear(in_features=32, out_features=32)

        self.out_fc = torch.nn.Linear(in_features=32, out_features=32)

        self.norm = torch.nn.LayerNorm(normalized_shape=32,
                                       elementwise_affine=True)

        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, Q, K, V, mask):
        b = Q.shape[0]
        # b phrases de taille 50, avec un embedding de taille 32
        # Q,K,V sont de taille b x 50 x 32

        clone_Q = Q.clone()
        # on garde une copie de Q pour la residual connection

        # on normalise Q,K,V
        Q = self.norm(Q)
        K = self.norm(K)
        V = self.norm(V)

        # on applique les transformations linéaires
        # 4 têtes d'attention, donc chaque tête d'attention aura taille 8
        # [b, 50, 32] -> [b, 4,50, 8]
        Q = Q.shape(b, 50, 4, 8).permute(0, 2, 1, 3)
        K = K.shape(b, 50, 4, 8).permute(0, 2, 1, 3)
        V = V.shape(b, 50, 4, 8).permute(0, 2, 1, 3)

        # on calcule l'attention
        score = attention(Q, K, V, mask)

        score = self.dropout(self.out_fc(score))

        score = score + clone_Q

        return score


