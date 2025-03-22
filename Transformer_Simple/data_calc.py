import random
import numpy as np
import torch

# Define the input dictionary (also used as the output dictionary),
# including only necessary tokens.
dict_x_calc = "<SOS>,<EOS>,<PAD>,0,1,2,3,4,5,6,7,8,9,a"
dict_x_calc = {word: i for i, word in enumerate(dict_x_calc.split(","))}
dict_xr_calc = [k for k, v in dict_x_calc.items()]

# For the addition task, we reuse the same dictionary for outputs.
# No uppercase conversion is needed here.
dict_y_calc = dict_x_calc.copy()
dict_yr_calc = dict_xr_calc.copy()


def get_data():
    """Generate a single addition data sample for a sequence-to-sequence task.

    This function randomly selects two number strings (s1 and s2) and computes their sum.
    The source sequence (x) is formed by concatenating s1, a separator 'a', and s2.
    The target sequence (y) is the digit-wise representation of the sum of s1 and s2.
    Both sequences are padded to fixed lengths and include <SOS> and <EOS> tokens.

    Returns:
        tuple:
            - x (torch.LongTensor): Encoded source sequence of shape [50].
            - y (torch.LongTensor): Encoded target sequence of shape [51].
    """
    # Define the digit vocabulary
    words = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

    # Define the probability of selecting each digit
    p = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    p = p / p.sum()

    # Randomly select a sequence of digits for s1
    n = random.randint(10, 20)
    s1 = np.random.choice(words, size=n, replace=True, p=p).tolist()

    # Randomly select a sequence of digits for s2
    n = random.randint(10, 20)
    s2 = np.random.choice(words, size=n, replace=True, p=p).tolist()

    # Convert s1 and s2 to integers and compute their sum
    y = int("".join(s1)) + int("".join(s2))
    y = list(str(y))

    # Create the source sequence x by concatenating s1, separator 'a', and s2
    x = s1 + ["a"] + s2

    # Add start (<SOS>) and end (<EOS>) tokens
    x = ["<SOS>"] + x + ["<EOS>"]
    y = ["<SOS>"] + y + ["<EOS>"]

    # Pad x to length 50 and y to length 51
    x = x + ["<PAD>"] * 50
    y = y + ["<PAD>"] * 51
    x = x[:50]
    y = y[:51]

    # Encode the sequences into token indices
    x = [dict_x_calc[i] for i in x]
    y = [dict_y_calc[i] for i in y]

    # Convert lists to tensors
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
        return get_data()


# Create a DataLoader for the addition dataset
loader_calc = torch.utils.data.DataLoader(
    dataset=Dataset(), batch_size=8, drop_last=True, shuffle=True, collate_fn=None
)
