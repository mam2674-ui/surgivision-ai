import torch
# Small fixtures should not oversubscribe CPU cores.
torch.set_num_threads(2)
