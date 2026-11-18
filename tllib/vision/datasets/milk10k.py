from torchvision.datasets import ImageFolder

class MILK10K(ImageFolder):
    CLASSES = ["akiec", "bcc", "bkl", "mel", "nv"]

    def __init__(self, root, split="train", transform=None, download=False):
        super(MILK10K, self).__init__(root=f"{root}/{split}", transform=transform)
        self.num_classes = len(self.CLASSES)
