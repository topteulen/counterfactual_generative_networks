import os

models = ["CNN","C8SteerableCNN","C8SteerableCNNSmall", "SO2SteerableCNN","SES","SES_V"]

datasets = ["colored_MNIST", "double_colored_MNIST", "wildlife_MNIST"]

additions = ['', '_counterfactual', '_counterfactual_rot', '_counterfactual_rot_scale', '_counterfactual_rot_scale_shear']


for model in models:
    for dataset in datasets:
        for addition in additions:
            os.system(f'python mnists/train_classifier.py --dataset {dataset}{addition} --model {model}')