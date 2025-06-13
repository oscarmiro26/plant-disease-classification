import torch
import torchvision.transforms as T

# Data augmentation methods, these should work well for CNNs
# These should be applied only to the training data, careful when using this later
train_augmentation_transforms = T.Compose(
    [
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=20),  # Moderate rotation
        T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    ]
)

# Normalization

# This one is very standard, but for now let's use the same normalization as ResNet so we can compare later
# normalize_transform = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]
normalize_transform = T.Normalize(mean=imagenet_mean, std=imagenet_std)

# We apply the augmentation techniqes to PIL data, convert to Tensor and then normalize
train_transforms = T.Compose(
    [train_augmentation_transforms, T.ToTensor(), normalize_transform]
)

# We also convert the validation and test set into tensors and normalize them, though we do not apply any augmentation
val_test_transforms = T.Compose([T.ToTensor(), normalize_transform])

print("Train transforms:", train_transforms)
print("Validation/Test transforms:", val_test_transforms)
