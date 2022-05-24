import os
import subprocess

models = ["CNN","C8SteerableCNN","C8SteerableCNNSmall", "SO2SteerableCNN","SES","SES_V"]

datasets = ["colored_MNIST", "double_colored_MNIST", "wildlife_MNIST"]

additions = ['', '_counterfactual', '_counterfactual_rot', '_counterfactual_rot_scale', '_counterfactual_rot_scale_shear']


for model in models[3:]:
    for dataset in datasets:
        for addition in additions:
            if dataset == "colored_MNIST" and "scale" not in addition and model == "SO2SteerableCNN":
                continue
            cmd = f'python mnists/train_classifier.py --dataset {dataset}{addition} --model {model}'
            print(f"\nRunning:\n {cmd}\n")
            result = subprocess.check_output(cmd, shell=True, text=True)
            with open(f'mnists/results/{model}_{dataset}{addition}_shell.txt', 'w') as f:
                f.write(result)
            f.close()
            print(result)
