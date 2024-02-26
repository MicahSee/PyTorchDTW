"""Classical DTW implemented using loops. Used as a point of comparison for accuracy
and execution time."""
import numpy as np

def compute_euclidean_distance_matrix(x, y) -> np.array:
    """Calculate distance matrix
    This method calcualtes the pairwise Euclidean distance between two sequences.
    The sequences can have different lengths.
    """

    if len(x) < len(y):
        x, y = y, x

    dist = np.zeros((len(x), len(y)))

    for i in range(len(x)):
        for j in range(len(y)):
            dist[i,j] = (x[i]-y[j])**2
    return dist

def compute_total_cost(cost_mat):
  up_cost_mat = cost_mat
  running_cost = cost_mat[0,0]
  x = 0
  y = 0

  cost_mat = np.pad(cost_mat, ((0,1), (0,1)), mode='constant', constant_values=np.inf)

  while True:
    if x == up_cost_mat.shape[1]-1 and y == up_cost_mat.shape[0]-1:
      break

    win = cost_mat[y:y+2, x:x+2].flatten()[1:]

    min_val = np.min(win[::-1])
    min_index = (2-np.argmin(win[::-1])) + 1

    x_adj = min_index % 2
    y_adj = min_index // 2

    running_cost += min_val

    x += x_adj
    y += y_adj

  return running_cost

def compute_accumulated_cost_matrix(x, y) -> np.array:
    """Compute accumulated cost matrix for warp path using Euclidean distance
    """
    distances = compute_euclidean_distance_matrix(x, y)

    # print(distances)

    # Initialization
    cost = np.zeros((len(y), len(x)))
    cost[0,0] = distances[0,0]

    # print(cost)

    for i in range(1, len(y)):
        cost[i, 0] = distances[i, 0] + cost[i-1, 0]

    for j in range(1, len(x)):
        cost[0, j] = distances[0, j] + cost[0, j-1]

    # print(cost)

    # Accumulated warp path cost
    for i in range(1, len(y)):
        for j in range(1, len(x)):
            cost[i, j] = min(
                cost[i-1, j],    # insertion
                cost[i, j-1],    # deletion
                cost[i-1, j-1]   # match
            ) + distances[i, j]

    return cost

def baseline_dtw(x, patterns):
    costs = np.zeros(patterns.shape[0])

    for i in range(patterns.shape[0]):
      cost_mat = compute_accumulated_cost_matrix(x, patterns[i])
      total_cost = compute_total_cost(cost_mat)
      costs[i] = total_cost

    return costs