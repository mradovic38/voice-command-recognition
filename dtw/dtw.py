import numpy as np
import scipy.spatial.distance as dist
from typing import Tuple, List

def dp(dist_mat: np.ndarray) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """
    Find the minimum-cost path through a distance matrix using dynamic programming.
    
    This function computes the optimal alignment path between two sequences by finding
    the path that minimizes the cumulative distance.
    
    Args:
        dist_mat (numpy.ndarray): A 2D array representing pairwise distances between 
                                  elements of two sequences.

    Returns:
        tuple: 
            - path (list of tuples): The optimal alignment path represented as a list of 
              index pairs (i, j).
            - cost_mat (numpy.ndarray): The cumulative cost matrix after alignment.
    """

    N, M = dist_mat.shape
    
    # Initialize cost matrix with infinity
    cost_mat = np.zeros((N + 1, M + 1))
    cost_mat[1:, 0] = np.inf
    cost_mat[0, 1:] = np.inf
    
    # Fill cost matrix with traceback information
    traceback_mat = np.zeros((N, M), dtype=int)
    for i in range(N):
        for j in range(M):
            penalty = [
                cost_mat[i, j],      # match (0)
                cost_mat[i, j + 1],  # insertion (1)
                cost_mat[i + 1, j]   # deletion (2)
            ]
            i_penalty = np.argmin(penalty)
            cost_mat[i + 1, j + 1] = dist_mat[i, j] + penalty[i_penalty]
            traceback_mat[i, j] = i_penalty
    
    # Traceback from bottom right
    i, j = N - 1, M - 1
    path = [(i, j)]
    while i > 0 or j > 0:
        tb_type = traceback_mat[i, j]
        # Match
        if tb_type == 0:
            i, j = i - 1, j - 1
        # Insertion
        elif tb_type == 1:
            i = i - 1
        # Deletion
        elif tb_type == 2:
            j = j - 1
        path.append((i, j))
    
    # Strip infinity edges from cost matrix
    cost_mat = cost_mat[1:, 1:]

    return (path[::-1], cost_mat)



def calculate_dtw_cost(seq1: np.ndarray, seq2: np.ndarray) -> float:
    """
    Calculate the normalized Dynamic Time Warping (DTW) cost between two sequences.
    
    DTW aligns two sequences by computing the optimal alignment path that minimizes 
    the distance between them. This implementation uses cosine distance as the 
    distance metric and dynamic programming for alignment.

    Args:
        seq1 (numpy.ndarray): Feature matrix of the first sequence
        seq2 (numpy.ndarray): Feature matrix of the second sequence

    Returns:
        float: The normalized DTW cost
    """

    dist_mat = dist.cdist(seq1, seq2, "cosine")
    _, cost_mat = dp(dist_mat)
    
    # Normalize alignment cost
    normalized_cost = cost_mat[-1, -1] / (seq1.shape[0] + seq2.shape[0])
    return normalized_cost
