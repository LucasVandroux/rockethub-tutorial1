#!/usr/bin/env python
import torch
from rockethub import Rocket
from PIL import Image
from sys import argv

# Select the image you want to test the Object Detection Model with
image_name = 'office'
# image_name = 'shop'
# image_name = 'street'

# Select the Rocket you want to test
rocket = "lucas/ssd"
# rocket = "igor/retinanet"
# rocket = "lucas/yolov3"

# Use arguments by the user if given
if len(argv) >= 2:

    image_name = argv[1]

    if len(argv) >= 3:
        rocket = argv[2]


# --- LOAD IMAGE ---
image_path = 'images/' + image_name + '.jpg'
img = Image.open(image_path)

# --- LOAD ROCKET ---
model = Rocket.land(rocket).eval()

# --- DETECTION ---
print("Using the rocket to do object detection on '" + image_path + "'...")
with torch.no_grad():
    img_tensor = model.preprocess(img)
    out = model(img_tensor)

print('Object Detection successful!')

# --- OUTPUT ---
# Print the output as a JSON
bboxes_out = model.postprocess(out, img)
print(len(bboxes_out), 'different objects were detected:')
print(*bboxes_out, sep='\n')

# Display the output over the image
img_out = model.postprocess(out, img, visualize=True)
img_out_path = 'out_' + image_name + '_' + rocket.replace('/', '_') + '.jpg'
img_out.save(img_out_path)
print("You can see the detections on the image: '" + img_out_path + "'.")
