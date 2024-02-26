import time
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from classical_dtw import baseline_dtw
from pytorch_dtw import DTW

def time_execution(f, args):
  start_time = time.time()

  val = f(*args)

  end_time = time.time()
  elapsed_time = (end_time - start_time) * 1000

  return elapsed_time, val

def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    dtw = DTW(device=device).to(device)

    n_patterns = [1, 10, 100, 200, 500, 1000, 5000, 10000]

    b_times = np.zeros(len(n_patterns))
    g_times = np.zeros(len(n_patterns))

    for i, n in tqdm(enumerate(n_patterns)):
        key = np.random.rand(50)
        patterns = np.random.rand(n,50)

        b_time, b_results = time_execution(baseline_dtw, (key, patterns))

        key_tensor = torch.Tensor(key).to(device).reshape(1, -1)
        patterns_tensor = torch.Tensor(patterns).to(device)

        g_time, g_results = time_execution(dtw, (key_tensor, patterns_tensor))

        b_times[i] = b_time
        g_times[i] = g_time

    ### Plot execution time comparison
    plt.figure()

    log_n_patterns = np.log10(n_patterns)

    plt.plot(log_n_patterns, b_times, label='Baseline DTW (CPU)')
    plt.plot(log_n_patterns, g_times, label='PyTorch DTW (GPU)')

    plt.xlabel('$log_{10}$(Number of Patterns)')
    plt.ylabel('Execution Time (ms)')

    plt.title('DTW Execution Time')

    plt.legend()