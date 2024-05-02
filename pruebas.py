import numpy as np
import torch

print(torch.cuda.device_count())
print(torch.cuda.get_device_name(torch.cuda.current_device))
