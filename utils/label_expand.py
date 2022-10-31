import torch
import numpy as np

def label_expand(labels1):
    labels0 = list()
    label_temp = labels1.to('cpu')
    label_temp = np.array(label_temp)
    label_temp = label_temp.tolist()
    for i in range(64):
        for j in range(10):
            labels0 = labels0.append(label_temp[i])
    labels0 = torch.tensor(labels1, device='cuda:0')
    return labels0