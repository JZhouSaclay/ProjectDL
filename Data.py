
# définir un dictionnaire pour les tokens
zidian_x = '<SOS>,<EOS>,<PAD>,0,1,2,3,4,5,6,7,8,9,q,w,e,r,t,y,u,i,o,p,a,s,d,f,g,h,j,k,l,z,x,c,v,b,n,m'
zidian_x = {word: i for i, word in enumerate(zidian_x.split(','))}
# enumerat permet de connaitre l'index de chaque élément

zidian_xr = [k for k, v in zidian_x.items()]
# on récupère que les clés du dictionnaire

zidian_y = {k.upper(): v for k, v in zidian_x.items()}
# on met les clés en majuscule

zidian_yr = [k for k, v in zidian_y.items()]
# Que les clés en majuscule


import random
import torch
import numpy as np

def get_data():
    words = [
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'q', 'w', 'e', 'r',
        't', 'y', 'u', 'i', 'o', 'p', 'a', 's', 'd', 'f', 'g', 'h', 'j', 'k',
        'l', 'z', 'x', 'c', 'v', 'b', 'n', 'm'
    ]
    p = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,
                  23,24,25,26])
    p = p / p.sum()
    # on crée un tableau de probabilité pour chaque lettre



    n = random.randint(30,48)
    # on choisit un nombre aléatoire entre 30 et 48
    x = np.random.choice(words, n, replace=True,p=p)
    # on choisit n lettres aléatoirement avec remise


    x = x.tolist()
    # on convertit en liste


    def f(i):
        i = i.upper()
        if not i.isdigit():
            return i
        i = 9 - int(i)
        return str(i)
    # on définit une fonction qui prend un élément i et le met en majuscule
    # si ce n'est pas un chiffre, on le retourne tel quel
    # si c'est un chiffre, on le transforme en 9 - i


    y = [f(i) for i in x]
    y = y + [y[-1]]
    # y = y + [y[-1]] permet de rajouter le dernier élément de y à y

    y = y[::-1]
    # on inverse y pour avoir un effet miroir


    x = ['<SOS>'] + x + ['<EOS>']
    y = ['<SOS>'] + y + ['<EOS>']


    x = x + ['<PAD>'] * 50
    y = y + ['<PAD>'] * 51
    # on rajoute des <PAD> pour avoir une taille de 50
    x = x[:50]
    y = y[:51]


    x = [zidian_x[i] for i in x]
    y = [zidian_y[i] for i in y]
    # on remplace chaque élément par son index dans le dictionnaire

    x = torch.tensor(x).long()
    y = torch.tensor(y).long()
    
    return x, y




# on crée une fonction qui génère des données
class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        super(Dataset, self).__init__()

    def __len__(self):
        return 10000
    
    def __getitem__(self, index):
        return get_data()
    

# on crée un DataLoader
loader = torch.utils.data.DataLoader(dataset=Dataset(), 
                                     batch_size=8,
                                     drop_last=True,
                                     shuffle=True,
                                     collate_fn=None)


