import argparse
import warnings
from tqdm import trange
import torch
import repackage
repackage.up()

# new
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from mnists.train_cgn import CGN
from mnists.dataloader import get_dataloaders
from utils import load_cfg
import os

def generate_cf_dataset(cgn, path, dataset_size, no_cfs, device, **kwargs):
    x, y = [], []
    cgn.batch_size = 100
    n_classes = 10

    total_iters = int(dataset_size // cgn.batch_size // no_cfs)
    for _ in trange(total_iters):

        # generate initial mask
        y_gen = torch.randint(n_classes, (cgn.batch_size,)).to(device)
        mask, _, _ = cgn(y_gen)

        # generate rotation angle
        transform = transforms.Compose([
            transforms.RandomAffine(**kwargs, interpolation=InterpolationMode.BILINEAR),
        ])

        mask_org = mask.clone()

        # generate counterfactuals, i.e., same masks, foreground/background vary
        for _ in range(no_cfs):
            with torch.no_grad():
                for i, m in enumerate(mask_org):
                    mask[i] = transform(m)

            _, foreground, background = cgn(y_gen, counterfactual=True)
            x_gen = mask * foreground + (1 - mask) * background

            x.append(x_gen.detach().cpu())
            y.append(y_gen.detach().cpu())

    dataset_y = torch.cat(y)
    print(f"x shape {len(x)*x[0].shape[0]}, y shape {dataset_y.shape}")
    torch.save([x, dataset_y], 'mnists/data/' + path)

def generate_dataset(dl, path):
    x, y = [], []
    for data in dl:
        x.append(data['ims'].cpu())
        y.append(data['labels'].cpu())

    dataset = [torch.cat(x), torch.cat(y)]

    print(f"Saving to {path}")
    print(f"x shape: {dataset[0].shape}, y shape: {dataset[1].shape}")
    torch.save(dataset, 'mnists/data/' + path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        choices=['colored_MNIST', 'double_colored_MNIST', 'wildlife_MNIST'],
                        help='Name of the dataset. Make sure the name and the weight_path match')
    parser.add_argument('--weight_path', default='',
                        help='Provide path to .pth of the model')
    parser.add_argument('--dataset_size', type=float, default=1e5,
                        help='Size of the dataset. For counterfactual data: the more the better.')
    parser.add_argument('--no_cfs', type=int, default=10,
                        help='How many counterfactuals to sample per datapoint')
    parser.add_argument('--affine_transform', choices=['', 'rot', 'rot_scale', 'rot_scale_shear'],
                        default='',
                        help='Provide a Affine transform that is applied to the mask')
    args = parser.parse_args()
    print(args)

    assert args.weight_path or args.dataset, "Supply dataset name or weight path."
    if args.weight_path: assert args.dataset, "Also supply the dataset type."

    # Generate the dataset
    if not args.weight_path:
        # get dataloader
        dl_train, dl_test = get_dataloaders(args.dataset, batch_size=1000, workers= 0 if os.name == "nt" else 12)
        # generate
        generate_dataset(dl=dl_train, path=f'{args.dataset}_train.pth')
        for name, dl in dl_test.items():
            generate_dataset(dl=dl, path=f'{args.dataset}_{name}.pth')

    # Generate counterfactual dataset
    else:
        # load model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cgn = CGN()
        cgn.load_state_dict(torch.load(args.weight_path, 'cpu'))
        cgn.to(device).eval()

        affine_choices = {
            ''                : {'degrees':0},
            'rot'             : {'degrees':180, 'translate':(0.1, 0.1)},
            'rot_scale'       : {'degrees':180, 'translate':(0.1, 0.1), 'scale':(0.5, 1.5)},
            'rot_scale_shear' : {'degrees':180, 'translate':(0.1, 0.1), 'scale':(0.5, 1.5), 'shear':30},
        }

        # generate
        print(f"Generating the counterfactual {args.dataset} of size {args.dataset_size}")
        generate_cf_dataset(cgn=cgn, path=f'{args.dataset}_counterfactual{"_" if args.affine_transform else ""}{args.affine_transform}.pth',
                            dataset_size=args.dataset_size, no_cfs=args.no_cfs,
                            device=device, **affine_choices[args.affine_transform])
