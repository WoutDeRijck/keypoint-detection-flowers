import os
import json
from PIL import Image, ImageDraw
"""
This converts bounding box json file to a center keypoint json file
"""

# assign directory
directory = '/home/nvidia/Documents/KeyPoints/keypoint-detection/flowers/dataset/Dataset55bda906-e605-4ca1-9345-f931af1c27cc'

# Final json file
dictionary = {"dataset": []}
# iterate over files in that directory
for i, filename in enumerate(sorted(os.listdir(directory))):
    f = os.path.join(directory, filename)
    # checking if it is a file and its a capture file
    if 'capture' in filename and os.path.isfile(f):
        print(filename + ': in process')

        file_json = open(f)
        data = json.load(file_json)

        

        for index, picture in enumerate(data["captures"]):
            bloemetjes = picture["annotations"][0]["values"]
            img = Image.open('/home/nvidia/Documents/KeyPoints/keypoint-detection/flowers/dataset/' + picture["filename"])
            
            img_x, img_y = img.size

            filePath = picture["filename"]
            newFilePath = filePath.replace('RGB03045d16-4f1c-4ae9-ba9f-fa5c1eccc572', 'images')
            # Make new folder with RGB png's
            # img_rgb = img.convert('RGB')
            # img_rgb.save('/home/nvidia/Documents/KeyPoints/keypoint-detection/flowers/dataset/dataset/' + newFilePath)
            new_dict = {"image_path" : newFilePath}
            center_keypoints = []

            for bloem in bloemetjes:
                x = bloem["x"]
                y = bloem["y"]
                width = bloem["width"]
                height = bloem["height"]
                center_x = (x+x+width)/2 / img_x
                center_y = (img_y - (y+y+height)/2) / img_y

                center_keypoints.append([center_x, center_y])
            new_dict["center_keypoints"] = center_keypoints
            dictionary["dataset"].append(new_dict)

# Serializing json 
json_object = json.dumps(dictionary, indent = 4)

# Writing to new json dataset
with open("/home/nvidia/Documents/KeyPoints/keypoint-detection/flowers/dataset/dataset/dataset.json", "w") as outfile:
    outfile.write(json_object)