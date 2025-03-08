import torch
from attention import attention


class MultiHeadAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_Q = torch.nn.Linear(32, 32)
        self.fc_K = torch.nn.Linear(32, 32)
        self.fc_V = torch.nn.Linear(32, 32)

        self.out_fc = torch.nn.Linear(32, 32)

        self.norm = torch.nn.LayerNorm(
            normalized_shape=32,
            elementwise_affine=True,
        )

        self.drouput = torch.nn.Dropout(0.1)

    def forward(self, Q, K, V, mask):
        # on va dire qu'on a b phrases de taille 50,
        # avec un embedding de taille 32
        # Q,K,V sont de taille b x 50 x 32
        b = Q.shape[0]

        # on duplicate Q pour utiliser dans la suite
        clone_Q = Q.clone()

        # normalisation
        Q = self.norm(Q)
        K = self.norm(K)
        V = self.norm(V)

        # on applique les transformations linéaires sans changer la taille
        Q = self.fc_Q(Q)
        K = self.fc_K(K)
        V = self.fc_V(V)

        # on divise en 4 têtes d'attention
        Q = Q.reshape(b, 50, 4, 8).permute(0, 2, 1, 3)
        K = K.reshape(b, 50, 4, 8).permute(0, 2, 1, 3)
        V = V.reshape(b, 50, 4, 8).permute(0, 2, 1, 3)

        # on calcule l'attention
        score = attention(Q, K, V, mask)

        # on calcule la sortie
        score = self.dropout(self.out_fc(score))

        score = clone_Q + score
        return score
