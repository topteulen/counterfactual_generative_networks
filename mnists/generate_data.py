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

class transform_to_masks():
    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, x):
        return (x > x.mean()*2 + self.threshold).int()

    def __repr__(self):
        return "Make the image a mask"

def generate_cf_dataset(cgn, path, dataset_size, no_cfs, device):
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
            transforms.RandomAffine(degrees=180, translate=(0.1, 0.1), scale=None, shear=None, interpolation=InterpolationMode.BILINEAR),
            transform_to_masks(0.2)
        ])

        with torch.no_grad():
            for i, m in enumerate(mask):
                mask[i] = transform(m)

        # generate counterfactuals, i.e., same masks, foreground/background vary
        for _ in range(no_cfs):
            _, foreground, background = cgn(y_gen, counterfactual=True)
            x_gen = mask * foreground + (1 - mask) * background

            x.append(x_gen.detach().cpu())
            y.append(y_gen.detach().cpu())

    dataset = [torch.cat(x), torch.cat(y)]
    print(f"x shape {dataset[0].shape}, y shape {dataset[1].shape}")
    torch.save(dataset, 'mnists/data/' + path)

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
    parser.add_argument('--dataset_size', type=float, default=5e4,
                        help='Size of the dataset. For counterfactual data: the more the better.')
    parser.add_argument('--no_cfs', type=int, default=1,
                        help='How many counterfactuals to sample per datapoint')
    args = parser.parse_args()
    print(args)

    assert args.weight_path or args.dataset, "Supply dataset name or weight path."
    if args.weight_path: assert args.dataset, "Also supply the dataset type."

    # Generate the dataset
    if not args.weight_path:
        # get dataloader
        dl_train, dl_test = get_dataloaders(args.dataset, batch_size=1000, workers=8)

        # generate
        generate_dataset(dl=dl_train, path=args.dataset + '_train.pth')
        generate_dataset(dl=dl_test, path=args.dataset + '_test.pth')

    # Generate counterfactual dataset
    else:
        # load model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cgn = CGN()
        cgn.load_state_dict(torch.load(args.weight_path, 'cpu'))
        cgn.to(device).eval()

        # generate
        print(f"Generating the counterfactual {args.dataset} of size {args.dataset_size}")
        generate_cf_dataset(cgn=cgn, path=args.dataset + '_counterfactual.pth',
                            dataset_size=args.dataset_size, no_cfs=args.no_cfs,
                            device=device)
