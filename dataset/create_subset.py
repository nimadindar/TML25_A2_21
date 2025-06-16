import torch
from torch.utils.data import Subset

def get_random_subset(dataset, subset_index: int, subset_size: int = 1000, seed: int = 42) -> Subset:
    """
    Splits a dataset into reproducible random subsets and returns the subset at the specified index.

    Args:
        dataset (Dataset): The full dataset.
        subset_index (int): The index of the subset to return (0-based).
        subset_size (int, optional): Number of elements in each subset. Defaults to 1000.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        Subset: A torch.utils.data.Subset containing `subset_size` elements from the dataset.
    """
    total_len = len(dataset)
    num_subsets = total_len // subset_size
    assert 0 <= subset_index < num_subsets, f"subset_index must be in [0, {num_subsets - 1}]"

    g = torch.Generator().manual_seed(seed)
    shuffled_indices = torch.randperm(total_len, generator=g)

    start = subset_index * subset_size
    end = start + subset_size
    subset_indices = shuffled_indices[start:end]

    return Subset(dataset, subset_indices.tolist())
