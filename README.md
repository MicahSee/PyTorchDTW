# PyTorchDTW
<<<<<<< HEAD
Implements the classical DTW algorithm in PyTorch, enabling multi-pattern matching and GPU acceleration

To run the DTW module on a GPU, you can use the code below:
`
from pytorch_dtw import DTW

if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")

dtw = DTW(device=device).to(device)
`

The DTW module currently supports matching one sequence against many other sequences.
The inputs to the forward() method must have shape (1, n) and (k, m), where n is the length of the sequence
to perform matching for, k is the number of patterns to match against, and m is the length of each pattern
being matched against (n and m can be different).

For example:
`
key = np.random.rand(1, 50)
patterns = np.random.rand(10, 75)

key_tensor = torch.Tensor(key).to(device)
patterns_tensor = torch.Tensor(patterns).to(device)

costs = dtw(key, patterns)
`
=======
Implements the classical DTW algorithm in PyTorch, enabling multi-pattern matching and GPU acceleration.
>>>>>>> 0ab5fd4af689adffeb9b6d2e13548f19f7cd6b77
