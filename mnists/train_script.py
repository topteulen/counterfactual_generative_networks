"""
Run this script to replicate all the results from the paper. 
This script will generate all the datasets which are used for training and
testing and will train all the models and save their results.
"""

import os
import subprocess


GENERATE_DATA = True


# Model configurations for the classifier.
models = ["CNN","C8SteerableCNNSmall", "C8SteerableCNN", "SO2SteerableCNN","SES","SES_V"]

# Variations of the MNIST dataset
datasets = ["colored_MNIST", "double_colored_MNIST", "wildlife_MNIST"]

# Counterfactual variations of each dataset.
additions = ['_counterfactual', '_counterfactual_rot', '_counterfactual_rot_scale', '_counterfactual_rot_scale_shear', '']

if not os.path.exists('mnists/data/colored_mnist/mnist_10color_jitter_var_0.020.npy'):
    print("download https://drive.google.com/u/0/uc?export=download&confirm=rHtT&id=1NSv4RCSHjcHois3dXjYw_PaLIoVlLgXu \nand unpack it in mnists/data")
    exit()

if not os.path.exists("mnists/data/colored_mnist/mnist_10color_double_testsets_jitter_var_0.02_0.025.npy"):
    os.system('python mnists/scripts/create_coloured_mninst.py')

for dataset in datasets:
    for addition in additions:
        if GENERATE_DATA:
            # Generate counterfactual train and test data
            if "counterfactual" in addition:
                affine_transform = '--affine_transform ' + '_'.join(addition.split('_')[2:]) if 'rot' in addition else ''
                cmd = f"python mnists/generate_data.py --weight_path mnists/experiments/cgn_{dataset}/weights/ckp.pth \
                --dataset {dataset} --no_cfs 100 --dataset_size 1000000 {affine_transform}"
            # Generate unmodified train and test data
            else:
                cmd = f'python mnists/generate_data.py --dataset {dataset}'

            print(f"execute {cmd}")
            os.system(cmd)

        for model in models:
            # Train the classifier.
            cmd = f'python mnists/train_classifier.py --dataset {dataset}{addition} --model {model} --epochs 2'
            file_loc = f'mnists/results/{model}_{dataset}{addition}_shell.txt'
            # Skip results file if it exists. Comment out to re-train.
            if os.path.exists(file_loc):
                print(f'skipping {cmd}')
                continue
            
            # Save shell output to file.
            print(f"\nRunning:\n {cmd}\n")
            result = subprocess.check_output(cmd, shell=True, text=True)
            with open(f'mnists/results/{model}_{dataset}{addition}_shell.txt', 'w') as f:
                f.write(result)
            f.close()
            print(result)
