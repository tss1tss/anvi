import requests
import json
import base64
import cv2 
from PIL import Image
import numpy as np
import io
# jso = {'name': 'tae', 'b': True}
# json_data = json.dumps(jso)
# data = {'param': json_data}


# Take in base64 string and return cv image
def stringToRGB(base64_string):
    imgdata = base64.b64decode(str(base64_string))
    img = Image.open(io.BytesIO(imgdata))
    opencv_img= cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
    return opencv_img 
# Open the file and send the request within the 'with' block
with open('/home/tss2tss/Pictures/63f2dff3f8f4cca3134d66e5f32fda2a.jpg', 'rb') as f:
    files = {'image': ('63f2dff3f8f4cca3134d66e5f32fda2a.png', f, 'image/jpeg')}
    # res = requests.post('http://localhost:3000/ros', data=data, files=files)
    res = requests.post('http://localhost:3000/ros', files=files)

    res = res.json()
    img_base64 = res['img']['data']
    img = stringToRGB(img_base64)
    image = base64.b64decode(img_base64)

cv2.imshow('window_name', img) 
cv2.waitKey(0) 
cv2.destroyAllWindows() 