import torch 
import math

class FullyConnectedOutput(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=32, out_features=64),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=64, out_features=32)
            torch.nn.Dropout(0.1)
            # Désactive aléatoirement 10% des neurones pour réduire l'overfitting
        )

        self.norm = torch.nn.LayerNorm(normalized_shape=32,
                                       elementwise_affine=True)
        # elementwise_affine=True indique que γ et paramètres de mise à l'échelle et de translation
        # sont apprenables, permettant un ajustement optimal pendant l'entraînement.


    # residual connection    
    def forward(self, x):
        clone_x = x.clone()

        x = self.norm(x)

        out = self.fc(x)

        out = out + clone_x

        return out