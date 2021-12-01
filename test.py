import torch
import os
import numpy as np
from model_build import model, processor
from skimage import io
import cv2

def loadImage(img_file):
    img = io.imread(img_file)           # RGB order
    if img.shape[0] == 2: img = img[0]
    if len(img.shape) == 2 : img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:   img = img[:,:,:3]
    img = np.array(img)

    return img

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']= ''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model.load_state_dict(torch.load('./finetune_27.pth', map_location='cpu'))
model.eval()
model.encoder.eval()
model.decoder.eval()
paths= './data/test2' ##### 수정
for i in os.listdir(paths):
    image = loadImage(os.path.join(paths,i))
    img = processor(image, return_tensors="pt").pixel_values
    generated_ids = model.generate(img)
    
    # text = processor.decode(generated_ids[0], skip_special_tokens=True)
    text = processor.decode(generated_ids[0])
    print(text)
