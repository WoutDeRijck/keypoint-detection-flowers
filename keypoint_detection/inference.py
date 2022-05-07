from keypoint_detection.models.detector import KeypointDetector

from torchvision import transforms as T
from PIL import Image


PATH = "/home/nvidia/Documents/KeyPoints_Wout/keypoint-detection-flowers/keypoint_detection/61e26c1d7cfe56a44584c186495e3153"

## Convert model to onnx
model = KeypointDetector.load_from_checkpoint(PATH)
model.eval()
# model.to_onnx("model_flower.onnx", tensor, export_params =True, opset_version=11)

## Convert png to tensor
img = Image.open("/home/nvidia/Documents/KeyPoints_Wout/keypoint-detection-flowers/flowers/dataset/dataset/images/rgb_10.png")
convert_tensor = T.ToTensor()
tensor = convert_tensor(img)
tensor = tensor.reshape(1, 3, 512, 512)

## Try inference
output = model(tensor)
output = output[0]
transform_toPIL = T.ToPILImage()
heatmap = transform_toPIL(output)
heatmap.show()