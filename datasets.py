import os
import torch
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from tqdm import tqdm
import torch.utils.data as data
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import TensorDataset
from torchvision.transforms.functional import rotate
from dataset.colored_mnist import COLORED_MNIST_PROTOCOL
import random
from PIL import ImageFilter

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def make_attr_labels(target_labels, bias_aligned_ratio):
    num_classes = target_labels.max().item() + 1
    num_samples_per_class = np.array([
            torch.sum(target_labels == label).item()
            for label in range(num_classes)
        ])

    ratios_per_class = bias_aligned_ratio * np.eye(num_classes) + (
        1 - bias_aligned_ratio
    ) / (num_classes - 1) * (1 - np.eye(num_classes))

    corruption_milestones_per_class = (
        num_samples_per_class[:, np.newaxis]
        * np.cumsum(ratios_per_class, axis=1)
    ).round()
    num_corruptions_per_class = np.concatenate([
            corruption_milestones_per_class[:, 0, np.newaxis],
            np.diff(corruption_milestones_per_class, axis=1),
        ], axis=1,)

    attr_labels = torch.zeros_like(target_labels)
    for label in range(10):
        indices = (target_labels == label).nonzero().squeeze()
        corruption_milestones = corruption_milestones_per_class[label]
        for corruption_idx, idx in enumerate(indices):
            attr_labels[idx] = np.min(
                np.nonzero(corruption_milestones > corruption_idx)[0]
            ).item()

    return attr_labels


class MultipleDomainDataset:
    N_STEPS = 5001           # Default, subclasses may override
    CHECKPOINT_FREQ = 100    # Default, subclasses may override
    N_WORKERS = 1            # Default, subclasses may override
    ENVIRONMENTS = None      # Subclasses should override
    INPUT_SHAPE = None       # Subclasses should override

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)

class MultipleEnvironmentMNIST(MultipleDomainDataset):
    def __init__(self, root, environments, dataset_transform, input_shape, num_classes):
        super().__init__()
        if root is None:
            raise ValueError('Data directory not specified!')

        original_dataset_tr = MNIST(root, train=True, download=False)
        original_dataset_te = MNIST(root, train=False, download=False)

        original_images = torch.cat((original_dataset_tr.data, original_dataset_te.data))
        original_labels = torch.cat((original_dataset_tr.targets, original_dataset_te.targets))

        shuffle = torch.randperm(len(original_images))

        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]

        self.datasets = []
        
        for i in range(len(environments)):
    
            # images = original_images[i::len(environments)]
            # labels = original_labels[i::len(environments)]
            
            images = original_images
            labels = original_labels
            color_labels = make_attr_labels(torch.LongTensor(labels), environments[i])
            # colored_img = protocol[color_label.item()](img, severity)
            # colored_img = np.moveaxis(np.uint8(colored_img), 0, 2)

            # images.append(colored_img)
            # attrs.append([target_label, color_label])
            domains = torch.empty((images.shape[0]))
            domains[:] = i
            self.datasets.append(dataset_transform(images, labels, environments[i], color_labels))

        self.input_shape = input_shape
        self.num_classes = num_classes


# modified from Learn from failure
class ColoredMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['+90%', '+80%', '-90%']

    def __init__(self, args):
        # previous [0.1, 0.2, 0.9], class_num = 2
        # now skew ratio [0.1, 0.995], class_num = 10
        self.class_num = 10
        self.input_shape = (3, 28, 28,)
        self.protocol = COLORED_MNIST_PROTOCOL
        self.severity = 1
        super(ColoredMNIST, self).__init__(args.data_path, [0.1, 0.1],
                                        self.color_lff_dataset, (3, 28, 28,), self.class_num)


    def color_lff_dataset(self, images, labels, environment, domains):
        x = np.zeros((len(images), 3, 28, 28))
        images = images.float()/255
        for bid, (img, target_label, color_label) in enumerate(zip(images, labels, domains)):
            
            colored_img = self.protocol[color_label.item()](img, self.severity)
            # colored_img = np.moveaxis(np.uint8(colored_img), 0, 2)
            x[bid] = colored_img
            # images.append(colored_img)
            # attrs.append([target_label, color_label])

        x = torch.from_numpy(x).float()
        y = labels.view(-1).long()
        domains = domains.view(-1)
        return TensorDataset(x, y, domains)

    def color_dataset(self, images, labels, environment, domains):
        # # Subsample 2x for computational convenience
        # images = images.reshape((-1, 28, 28))[:, ::2, ::2]
        # Assign a binary label based on the digit
        labels = (labels < 5).float()
        # Flip label with probability 0.25
        labels = self.torch_xor_(labels, self.torch_bernoulli_(0.25, len(labels)))

        # Assign a color based on the label; flip the color with probability e
        colors = self.torch_xor_(labels, self.torch_bernoulli_(environment, len(labels)))
        images = torch.stack([images, images], dim=1)

        # Apply the color to the image by zeroing out the other color channel
        images[torch.tensor(range(len(images))), (1 - colors).long(), :, :] *= 0

        x = images.float().div_(255.0)
        y = labels.view(-1).long()

        # domain = domains.view(-1).long()
        # return TensorDataset(x, y, domain)
    
        return TensorDataset(x, y)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()


class MultiViewDataInjector(object):
    def __init__(self, *args):
        self.transforms = args[0]
        self.random_flip = transforms.RandomHorizontalFlip()

    def __call__(self, sample, *with_consistent_flipping):
        if with_consistent_flipping:
            sample = self.random_flip(sample)
        output = [transform(sample) for transform in self.transforms]
        return output

class RotatedMNIST(MultipleEnvironmentMNIST):
    # ENVIRONMENTS = ['0', '15', '30', '45', '60', '75']
    def __init__(self, args):
        self.environments = [0, 15, 30, 45, 60, 75]
        self.img_size = 28
        self.num_classes = 10
        self.test_env = args.test_envs
        super(RotatedMNIST, self).__init__(args.data_path, self.environments, self.rotate_dataset,
                                           (1, self.img_size, self.img_size,), self.num_classes)

    def rotate_dataset(self, images, labels, angle, domains):
        # angles = [0, 15, 30, 45, 60, 75]
        # angles.remove(angles[self.test_env[0]])

        rotation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Lambda(lambda x: rotate(x, angle, fill=(0,), resample=Image.BICUBIC)),
            transforms.ToTensor()])
        x = torch.zeros(len(images), 1, 28, 28)

        for i in range(len(images)):
            x[i] = rotation(images[i])
        y = labels.view(-1)
        domains = domains.view(-1)
        return TensorDataset(x, y, domains)


class MultipleEnvironmentImageFolder(MultipleDomainDataset):
    def __init__(self, root, test_envs, augment, args):
        super().__init__()
        environments = [f.name for f in os.scandir(root) if f.is_dir()]
        environments = sorted(environments)

        # MAE pretrain simple augmentation
        # if args.recon_mission:
        #     transform = transforms.Compose([
        #                 transforms.RandomResizedCrop(args.input_size, scale=(0.4, 1.0), interpolation=3),  # 3 is bicubic
        #                 transforms.RandomHorizontalFlip(),
        #                 transforms.ToTensor(),
        #                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        #             )

        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        data_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        if args.recon_mission:
            augment_transform = transforms.Compose([
                            transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
                        )
        else:
            augment_transform = transforms.Compose([
                # CycleMAE 
                # transforms.RandomResizedCrop(224, interpolation=3),
                # transforms.RandomHorizontalFlip(),
                # transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

                # DDG
                transforms.Resize((224,224)),
                transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
                transforms.RandomGrayscale(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                
                # MOCO V3
                # transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
                # transforms.RandomApply(
                #     [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8  # not strengthened
                # ),
                # transforms.RandomGrayscale(p=0.2),
                # transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
                # transforms.RandomHorizontalFlip(),
                # transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        self.datasets = []
        self.labels = []
        for i, environment in enumerate(environments):

            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform

            path = os.path.join(root, environment)
            env_dataset = ImageFolder(
                # path, transform=env_transform, 
                path, transform=MultiViewDataInjector([data_transform, env_transform]),
                sample_pos=args.sample_pos,
                is_mix = args.is_mix
                )

            self.datasets.append(env_dataset)
            self.labels.append(env_dataset.targets)

        self.input_shape = (3, 224, 224,)
        self.num_classes = len(self.datasets[-1].classes)

class PACS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = [ "A", "C", "P", "S"]
    def __init__(self, args):
        # args.data_path, args.test_env
        self.dir = os.path.join(args.data_path, "PACS/")
        super().__init__(self.dir, args.test_envs, args.augment, args)


class VLCS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["C", "L", "S", "V"]
    def __init__(self, args):
        self.dir = os.path.join(args.data_path, "VLCS/")
        super().__init__(self.dir, args.test_envs, args.augment, args)

class DomainNet(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 1000
    ENVIRONMENTS = ["clip", "info", "paint", "quick", "real", "sketch"]
    def __init__(self, args):
        self.dir = os.path.join(args.data_path, "udg_dn/")
        super().__init__(self.dir, args.test_envs, args.augment, args)

class OfficeHome(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["A", "C", "P", "R"]
    def __init__(self, args):
        self.dir = os.path.join(args.data_path, "OfficeHome/")
        super().__init__(self.dir, args.test_envs, args.augment, args)


class NICO(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["sheep", "rat", "monkey", "horse", "elephant", "dog",
                    "cow", "cat", "bird", "bear"]
    # ENVIRONMENTS = ['airplane','bicycle','boat','bus','car','helicopter','motorcycle']

    def __init__(self, args):
        # args.data_path, args.test_env
        self.dir = os.path.join(args.data_path, "Animal/")
        super().__init__(self.dir, args.test_envs, args.augment, args)


class VisionDataset(data.Dataset):
    _repr_indent = 4

    def __init__(self, root, transforms=None, transform=None, target_transform=None):
        if isinstance(root, torch._six.string_classes):
            root = os.path.expanduser(root)
        self.root = root

        has_transforms = transforms is not None
        has_separate_transform = transform is not None or target_transform is not None
        if has_transforms and has_separate_transform:
            raise ValueError("Only transforms or transform/target_transform can "
                             "be passed as argument")

        # for backwards-compatibility
        self.transform = transform
        self.target_transform = target_transform

        if has_separate_transform:
            transforms = StandardTransform(transform, target_transform)
        self.transforms = transforms

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __repr__(self):
        head = "Dataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        if self.root is not None:
            body.append("Root location: {}".format(self.root))
        body += self.extra_repr().splitlines()
        if hasattr(self, "transforms") and self.transforms is not None:
            body += [repr(self.transforms)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)

    def _format_transform_repr(self, transform, head):
        lines = transform.__repr__().splitlines()
        return (["{}{}".format(head, lines[0])] +
                ["{}{}".format(" " * len(head), line) for line in lines[1:]])

    def extra_repr(self):
        return ""


class StandardTransform(object):
    def __init__(self, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform

    def __call__(self, input, target):
        if self.transform is not None:
            input = self.transform(input)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return input, target

    def _format_transform_repr(self, transform, head):
        lines = transform.__repr__().splitlines()
        return (["{}{}".format(head, lines[0])] +
                ["{}{}".format(" " * len(head), line) for line in lines[1:]])

    def __repr__(self):
        body = [self.__class__.__name__]
        if self.transform is not None:
            body += self._format_transform_repr(self.transform,
                                                "Transform: ")
        if self.target_transform is not None:
            body += self._format_transform_repr(self.target_transform,
                                                "Target transform: ")

        return '\n'.join(body)


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_dataset(directory, class_to_idx, extensions=None, is_valid_file=None):
    instances = []
    directory = os.path.expanduser(directory)
    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = path, class_index
                    instances.append(item)
    return instances


class DatasetFolder(VisionDataset):

    def __init__(self, root, loader, extensions=None, transform=None,
                 target_transform=None, is_valid_file=None, sample_pos=False, is_mix=False):
        super(DatasetFolder, self).__init__(root, transform=transform, target_transform=target_transform)
        
        classes, class_to_idx = self._find_classes(self.root)
        samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.sample_pos = sample_pos
        self.is_mix = is_mix
        self.targets = np.array([s[1] for s in samples])

        if sample_pos:
            class_inds = [np.argwhere(self.targets==i) for i in range(len(self.classes))]
            self.pos_inds = [class_inds[self.targets[ind]][np.random.randint(len(class_inds[self.targets[ind]]))].item() for ind in range(len(self.targets))]

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.sample_pos:
            path, pos = self.samples[self.pos_inds[index]]
            pos_sample = self.loader(path)
            if self.transform is not None:
                pos_sample = self.transform(pos_sample)
            return sample, target, pos_sample
        return sample, target

    def __len__(self):
        return len(self.samples)


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImageFolder(DatasetFolder):
    """
        A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, is_valid_file=None, sample_pos=False, is_mix=False):
        super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file,
                                          sample_pos=sample_pos,
                                          is_mix = is_mix)
        self.imgs = self.samples
