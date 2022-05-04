import json
from PIL import Image, ImageDraw

file_json = open('/home/nvidia/Documents/dataset/synth_20k_v3/Dataset55bda906-e605-4ca1-9345-f931af1c27cc/captures_000.json')
data = json.load(file_json)

for index, picture in enumerate(data["captures"]):
    if index == 0:
        bloemetjes = picture["annotations"][0]["values"]
        img = Image.open('/home/nvidia/Documents/dataset/synth_20k_v3/RGB03045d16-4f1c-4ae9-ba9f-fa5c1eccc572/rgb_2.png')
        for bloem in bloemetjes:
            x = bloem["x"]
            y = bloem["y"]
            width = bloem["width"]
            height = bloem["height"]
            shape = [(x, y), (x + width, y + height)]
            img1 = ImageDraw.Draw(img) 
            img1.rectangle(shape, outline ="blue")
        img.show()
        img.save("flowers.png")