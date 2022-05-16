import torch

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
    def __init__(self, threshold=0.2):
        self.threshold = threshold

    def __call__(self, x):
        """ Depending on the scaling (thickness of a digit scales)
            the threshold should change, a larger the digit should have a higher threshold
            to counter the scaling of the thickness of the digit.
            This is especcially needed for digits that are downscaled """
        color = torch.amax(x, (0,1))
        gray = torch.sum(x, 2)
        max_gray = torch.sum(color)
        mask = (gray > max_gray*self.threshold + gray.mean()*2).float()
        return mask.reshape(*mask.shape, 1) * color.reshape(1, -1)

    def __repr__(self):
        return "Make the image into a mask"
