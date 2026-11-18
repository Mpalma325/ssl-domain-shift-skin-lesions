from torchvision.datasets import ImageFolder

class HAM10000(ImageFolder):
    CLASSES = ["akiec", "bcc", "bkl", "mel", "nv"]

    def __init__(self, root, split="train", transform=None, download=False):
        super(HAM10000, self).__init__(root=f"{root}/{split}", transform=transform)
        self.num_classes = len(self.CLASSES)
