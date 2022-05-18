import torch
import matplotlib.pyplot as plt

class transform_to_masks():
    def __init__(self, threshold=0.2):
        self.threshold = threshold

    def __call__(self, x):
        """ Depending on the scaling (thickness of a digit scales)
            the threshold should change, a larger the digit should have a higher threshold
            to counter the scaling of the thickness of the digit.
            This is especcially needed for digits that are downscaled """
        return (x > x.mean()*2 + self.threshold).int()

    def __repr__(self):
        return "Make the image into a mask"

class transform_to_one_color():
    def __init__(self, threshold=0.1, eps=1e-5):
        self.threshold = threshold
        self.eps = eps

    def __call__(self, x):
        """ Depending on the scaling (thickness of a digit scales)
            the threshold should change, a larger the digit should have a higher threshold
            to counter the scaling of the thickness of the digit.
            This is especcially needed for digits that are downscaled """
        color = torch.amax(x, (1,2))
        print(color.shape)
        gray = torch.mean(x, 0)
        # plt.imshow(torch.mul((gray > -1 + self.eps), gray), cmap='gray')
        # plt.show()
        max_gray = torch.mean(color)
        mask = (gray > ((gray > -1 + self.eps) * gray).mean()).float()
        print(mask.shape, gray.shape, max_gray, gray.mean(), gray.mean() + max_gray*self.threshold, ((gray > -1 + self.eps) * gray).mean())
        a = mask.reshape(*mask.shape, 1) * color.reshape(1, -1)
        print(a.shape)
        plt.imshow(mask, cmap='gray')
        plt.show()
        # plt.imshow(a)
        # plt.show()
        return torch.moveaxis(a, 2, 0)

    def __repr__(self):
        return "Make the image into a mask"


class test():
    def __init__(self, threshold=0.1):
        self.threshold = threshold

    def __call__(self, x):
        """ Depending on the scaling (thickness of a digit scales)
            the threshold should change, a larger the digit should have a higher threshold
            to counter the scaling of the thickness of the digit.
            This is especcially needed for digits that are downscaled """
        plt.imshow(x.mean(0), cmap='gray')
        plt.show()
        return x

    def __repr__(self):
        return "Make the image into a mask"
