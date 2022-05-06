import torch
import torch.nn as nn
import torch.optim as optim
from keypoint_detection.models.detector import KeypointDetector
from keypoint_detection.models.backbones.backbone_factory import BackboneFactory
from keypoint_detection.models.loss import LossFactory

from keypoint_detection.models.backbones.unet import UnetBackbone

from torchvision import transforms
from PIL import Image

## Convert png to tensor
img = Image.open("/home/nvidia/Documents/KeyPoints_Wout/keypoint-detection-flowers/flowers/dataset/dataset/images/rgb_10.png")
convert_tensor = transforms.ToTensor()
tensor = convert_tensor(img)
tensor = tensor.reshape(1, 3, 512, 512)

PATH = "/home/nvidia/Documents/KeyPoints_Wout/keypoint-detection-flowers/keypoint_detection/61e26c1d7cfe56a44584c186495e3153"

## TEST
# Backbone = BackboneFactory.create_backbone("Unet", n_channels_in=3, n_downsampling_layers=2, n_resnet_blocks=3, n_channels=16, kernel_size=3)
# loss = LossFactory.create_loss(loss="bce")
# model = KeypointDetector(2, "2 4", 1, 3e-4, ap_epoch_freq=10, ap_epoch_start=10, lr_scheduler_relative_threshold=0.0, keypoint_channels="center_keypoints", backbone=Backbone, loss_function=loss)
# model.eval()
#
#
# img  = torch.randn(1,3,512,512)
# print("execution test")
# print(model)
# print(model(torch.rand(1,3,512,512)))
# model.to_onnx("model_flower.onnx", torch.randn(1,3,512,512), export_params =True, opset_version=11)

## Try inference
model = KeypointDetector.load_from_checkpoint(PATH)
model.eval()

# model.to_onnx("model_flower.onnx", tensor, export_params =True, opset_version=11)
heatmap = model(img)