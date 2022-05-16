import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    # color_var needs to be in the range 0.020 to 0.050 incremented with 0.005
    color_var = [0.02, 0.025]
    data_dic = []
    for c in color_var:
        data_path = 'mnists/data/colored_mnist/mnist_10color_jitter_var_%.03f.npy'%c
        data_dic.append(np.load(data_path, encoding='latin1', allow_pickle=True).item())

    original_data = []
    original_labels = []
    counterfactual_data = []
    counterfactual_labels = []
    for d in data_dic:
        original_data.append(d['train_image'])
        original_labels.append(d['train_label'])

    for d in data_dic:
        halfway_index = len(d['test_image']) // 2
        counterfactual_data.append(d['test_image'][:halfway_index])
        counterfactual_labels.append(d['test_label'][:halfway_index])

    original_data = np.concatenate(original_data)
    original_labels = np.concatenate(original_labels)

    counterfactual_data = np.concatenate(counterfactual_data)
    counterfactual_labels = np.concatenate(counterfactual_labels)

    train_data, test_data, train_labels, test_labels = train_test_split(original_data, original_labels, test_size=0.5)

    # test data is now 60000 instead of the 10000 so it should reduced to the size of counterfactual_data
    test_data = test_data[:len(counterfactual_data)]
    test_labels = test_labels[:len(counterfactual_data)]

    data_dict = {'train_image'          : train_data,
                 'train_label'          : train_labels,
                 'test_image'           : test_data,
                 'test_label'           : test_labels,
                 'counterfactual_image' : counterfactual_data,
                 'counterfactual_label' : counterfactual_labels}

    np.save('/'.join(data_path.split('/')[:-1] + ['mnist_10color_double_testsets_jitter_var_0.02+0.025.npy']), data_dict, allow_pickle=True)
