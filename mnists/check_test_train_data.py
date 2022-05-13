import numpy as np
import matplotlib.pyplot as plt
from torch import tensor
import torch

if __name__ == '__main__':
    # check train or test dataset
    train = True

    # color_var needs to be in the range 0.020 to 0.050 incremented with 0.005
    color_var = np.arange(0.020, 0.051, 0.005)
    data_dic = []
    for c in color_var:
        data_path = 'mnists/data/colored_mnist/mnist_10color_jitter_var_%.03f.npy'%c
        data_dic.append(np.load(data_path, encoding='latin1', allow_pickle=True).item())

    ims = []
    labels = []
    for d in data_dic:
        ims.append(d['train_image'])
        labels.append(tensor(d['train_label'], dtype=torch.long))

    for d in data_dic:
        ims.append(d['test_image'])
        labels.append(tensor(d['test_label'], dtype=torch.long))

    for i, img in enumerate(zip(*ims)):
        f, axarr = plt.subplots(2,len(color_var))
        for j in range(2*len(color_var)):
            axarr[j//len(color_var), j%len(color_var)].imshow(img[j])
        plt.show()
