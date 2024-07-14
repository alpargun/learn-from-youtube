import torch
import torchvision.transforms as transforms
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample
)

class VideoTransform(object):

    def __init__(
        self,
        normalize=((0.485, 0.456, 0.406), # ImageNet mean and std
                   (0.229, 0.224, 0.225))
    ):
        self.side_size = 256
        self.normalize = normalize
        self.crop_size = 256

        self.mean_ = torch.tensor(self.normalize[0], dtype=torch.float32)
        self.std_ = torch.tensor(self.normalize[1], dtype=torch.float32)
        
        # Scale PIL and tensor conversions uint8 space by 255.
        #self.mean *= 255.
        #self.std *= 255.


    def __call__(self, buffer):

        # buffer = torch.tensor(buffer, dtype=torch.float32)

        buffer = buffer.permute(3, 0, 1, 2)  # T H W C -> C T H W

        #buffer = _tensor_normalize_inplace(buffer, self.mean, self.std)

        transform=Compose(
            [
                Lambda(lambda x: x/255.0), # scale between [0,1]
                NormalizeVideo(self.mean_, self.std_),
                ShortSideScale(
                    size=self.side_size
                ),
                CenterCropVideo(crop_size=(self.crop_size, self.crop_size))
            ]
        )
        
        buffer = transform(buffer)

        buffer = buffer.permute(1, 2, 3, 0) # C T H W -> T H W C

        return buffer

    def img_tensor_denormalize(self, img):
        """
        De-normalize a given tensor by multiplying with the std and adding the mean.
        Args:
            tensor (tensor): tensor to normalize.
            mean (tensor or list): mean value to subtract.
            std (tensor or list): std to divide.
        """
        img = img * self.std_
        img = img + self.mean_
        img = img * 255.
        
        return img


def tensor_normalize(tensor, mean, std):
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize.
        mean (tensor or list): mean value to subtract.
        std (tensor or list): std to divide.
    """
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()
        tensor = tensor / 255.0
    if type(mean) == list:
        mean = torch.tensor(mean)
    if type(std) == list:
        std = torch.tensor(std)
    tensor = tensor - mean
    tensor = tensor / std
    return tensor


def _tensor_normalize_inplace(tensor, mean, std):
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize (with dimensions C, T, H, W).
        mean (tensor): mean value to subtract (in 0 to 255 floats).
        std (tensor): std to divide (in 0 to 255 floats).
    """
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()

    C, T, H, W = tensor.shape
    tensor = tensor.view(C, -1).permute(1, 0)  # Make C the last dimension
    tensor.sub_(mean).div_(std)
    tensor = tensor.permute(1, 0).view(C, T, H, W)  # Put C back in front
    return tensor