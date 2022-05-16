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
