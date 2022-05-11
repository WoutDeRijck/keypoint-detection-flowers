from keypoint_detection.models.detector import KeypointDetector
import keypoint_detection.utils.heatmap as hm
from torchvision import transforms as T
from PIL import Image

import torch
import torchvision

PATH = "./flowers/61e26c1d7cfe56a44584c186495e3153"

## Convert model to onnx
model = KeypointDetector.load_from_checkpoint(PATH)
model.eval()
# model.to_onnx("model_flower2.onnx", tensor, export_params =True, opset_version=11)

## Convert png to tensor
img = Image.open("./flowers/test-flowers/goudsbloem-min.png")
img = img.convert('RGB')
img = img.resize((512, 512))
img.save("./flowers/inference/input.png")

convert_tensor = T.ToTensor()
tensor = convert_tensor(img)
tensor = tensor.reshape(1, 3, 512, 512)

## Try inference
output = model.forward(tensor)
output = output[0]
transform_toPIL = T.ToPILImage()
heatmap = transform_toPIL(output)
heatmap.save("./flowers/inference/heatmap.png")

transform = torchvision.transforms.ToTensor()
overlay = transform(hm.overlay_image_with_heatmap(tensor[0], output.cpu().detach(), alpha=0.1))
overlay_im = transform_toPIL(overlay)
overlay_im.save("./flowers/inference/overlay.png")
overlay_im.show()