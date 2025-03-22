import random
import numpy as np
import torch

# Define dictionary mapping for source tokens
dict_x = "<SOS>,<EOS>,<PAD>,0,1,2,3,4,5,6,7,8,9,q,w,e,r,t,y,u,i,o,p,a,s,d,f,g,h,j,k,l,z,x,c,v,b,n,m"
dict_x = {word: i for i, word in enumerate(dict_x.split(","))}
dict_xr = [k for k, v in dict_x.items()]

# Define dictionary mapping for target tokens (uppercase version of source tokens)
dict_y = {k.upper(): v for k, v in dict_x.items()}
dict_yr = [k for k, v in dict_y.items()]


def get_data_translation():
    """Generate a translation data pair for sequence transduction.

    This function creates a random sequence of characters based on a predefined vocabulary,
    applies a transformation (converting letters to uppercase and digits to their complementary
    value within 10), and then pads and encodes the sequence with start and end tokens.

    Returns:
        tuple: A tuple (x, y) where:
            - x (torch.LongTensor): The source sequence tensor of shape [seq_length] (padded).
            - y (torch.LongTensor): The target sequence tensor of shape [seq_length] (padded).
    """
    # Define vocabulary for translation
    words = [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "q",
        "w",
        "e",
        "r",
        "t",
        "y",
        "u",
        "i",
        "o",
        "p",
        "a",
        "s",
        "d",
        "f",
        "g",
        "h",
        "j",
        "k",
        "l",
        "z",
        "x",
        "c",
        "v",
        "b",
        "n",
        "m",
    ]

    # Define the probability for each word to be selected
    p = np.array(
        [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
        ]
    )
    p = p / p.sum()

    # Randomly choose n words
    n = random.randint(30, 48)
    x = np.random.choice(words, size=n, replace=True, p=p).tolist()

    # Transform x to generate y:
    # Convert letters to uppercase; for digits, use the complementary number (9 - digit)
    def f(i):
        i = i.upper()
        if not i.isdigit():
            return i
        return str(9 - int(i))

    y = [f(i) for i in x]
    # Duplicate the last element and then reverse the list
    y = y + [y[-1]]
    y = y[::-1]

    # Add start and end tokens to both sequences
    x = ["<SOS>"] + x + ["<EOS>"]
    y = ["<SOS>"] + y + ["<EOS>"]

    # Pad sequences to a fixed length
    x = x + ["<PAD>"] * 50
    y = y + ["<PAD>"] * 51
    x = x[:50]
    y = y[:51]

    # Encode tokens using the defined dictionaries
    x = [dict_x[i] for i in x]
    y = [dict_y[i] for i in y]

    # Convert lists to torch LongTensors
    x = torch.LongTensor(x)
    y = torch.LongTensor(y)
    return x, y


class Dataset(torch.utils.data.Dataset):
    """Custom dataset for the addition task.

    This dataset provides samples where two digit strings are randomly generated
    and summed. The input sequence is the concatenation of both digit strings,
    and the target sequence is the digit-wise representation of their sum.
    """

    def __init__(self):
        """Initialize the addition dataset."""
        super(Dataset, self).__init__()

    def __len__(self):
        """Return the number of samples in the dataset.

        Returns:
            int: The length of the dataset (fixed at 100000).
        """
        return 100000

    def __getitem__(self, i):
        """Retrieve the i-th sample from the dataset.

        Args:
            i (int): Index of the sample.

        Returns:
            tuple:
                - x (torch.LongTensor): Source sequence tensor of shape [50].
                - y (torch.LongTensor): Target sequence tensor of shape [51].
        """
        return get_data_translation()


# Create a DataLoader for the translation dataset
loader_trans = torch.utils.data.DataLoader(
    dataset=Dataset(), batch_size=8, drop_last=True, shuffle=True, collate_fn=None
)
