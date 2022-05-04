import torch
import torch.nn as nn
import torch.optim as optim
from models.detector import KeypointDetector
from models.backbones.backbone_factory import BackboneFactory
from models.loss import LossFactory

from models.backbones.unet import UnetBackbone

PATH = "/home/nvidia/Downloads/3c9503bd46aca5ac709f499c5f95a821"


Backbone = BackboneFactory.create_backbone("Unet", n_channels_in=3, n_downsampling_layers=2, n_resnet_blocks=3, n_channels=16, kernel_size=3)
loss = LossFactory.create_loss(loss="bce")


model = KeypointDetector(2, "2 4", 1, 3e-4, ap_epoch_freq=10, ap_epoch_start=10, lr_scheduler_relative_threshold=0.0, keypoint_channels="center_keypoints", backbone=Backbone, loss_function=loss)
# model = KeypointDetector.load_from_checkpoint(PATH)
model.eval()
print("execution test")
print(model)
print(model(torch.rand(1,3,512,512)))
model.to_onnx("test.onnx", torch.randn(1,3,512,512), export_params =True, opset_version=11)
img  = torch.randn(1,3,512,512)
heatmap = model.forward(img)

print(heatmap.shape)