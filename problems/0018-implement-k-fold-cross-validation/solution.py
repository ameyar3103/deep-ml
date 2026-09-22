import numpy as np
from typing import List, Tuple

def k_fold_cross_validation(n_samples: int, k: int = 5, shuffle: bool = True) -> List[Tuple[List[int], List[int]]]:
    """
    Generate train/test index splits for k-fold cross-validation.
    
    Args:
        n_samples: Total number of samples in the dataset
        k: Number of folds (default 5)
        shuffle: Whether to shuffle indices before splitting (default True)
    
    Returns:
        List of (train_indices, test_indices) tuples
    """
    # Your code here
    values = []
    for i in range(n_samples):
        values.append(i)
    
    if shuffle:
        np.random.shuffle(values)
    
    folds = np.array_split(values, k)
    ans = []
    for i in range(k):
        test_indices = folds[i].tolist()
        train_indices = []

        for j in range(k):
            if j != i:
                train_indices.extend(folds[j].tolist())

        ans.append((train_indices, test_indices))

    return ans
    
    pass