import torch
import os
import numpy as np
from model_build import model, processor
from skimage import io
from custom_dataset import *
import cv2

def loadImage(img_file):
    img = io.imread(img_file)           # RGB order
    if img.shape[0] == 2: img = img[0]
    if len(img.shape) == 2 : img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:   img = img[:,:,:3]
    img = np.array(img)

    return img

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']= '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.load_state_dict(torch.load('./deploy_model_40.pth', map_location='cpu'))
model.eval()
model.encoder.eval()
model.decoder.eval()
test_df = CustomDataset(None,processor=processor, data_dir='./data/test_data/').df
eval_dataset = CustomDataset(test_df,processor=processor, data_dir='./data/test_data/')
eval_loader = torch.utils.data.DataLoader(
    	dataset=eval_dataset,
       batch_size=128,
       shuffle=False,            
       num_workers=4,
       pin_memory=True)
# paths= './data/test2'
# for i in os.listdir(paths):
    # image = loadImage(os.path.join(paths,i))
    # img = processor(image, return_tensors="pt").pixel_values
    # generated_ids = model.generate(img)
    
    # # text = processor.decode(generated_ids[0], skip_special_tokens=True)
    # text = processor.decode(generated_ids[0])
    # print(text)
    #평가

def levenshtein(s1, s2, debug=False):
    if len(s1) < len(s2):
        return levenshtein(s2, s1, debug)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))

        if debug:
            print(current_row[1:])

        previous_row = current_row
    return previous_row[-1]
total = []
import re
import time
st = time.time()
for ii, batch in enumerate(eval_loader):
    print(ii)
    model.to(device)
    img_batch = batch['pixel_values'].to(device)
    labels_batch = batch['labels']
    text_label = batch['text']
    with torch.no_grad():
        outputs = model.generate(img_batch,max_length=4)

        preds = processor.tokenizer.batch_decode(outputs)
        for index,test in enumerate(preds):
            sep_idx = test.find('[SEP]')
            if test.startswith('[CLS]'):
                # print(text_label[index])
                cls_idx = test.find('[CLS]')
                pred = test[cls_idx+len('[CLS]'):sep_idx] if not (sep_idx == -1) else test[cls_idx+len('[CLS]'):]
                pred = re.sub('[CLS]|[SEP]','',pred)
                total.append(text_label[index] +'--'+pred)
            else:
                cls_idx = 0
                pred = test[:sep_idx] if not (sep_idx == 1) else test
                pred = re.sub('[CLS]|[SEP]','',pred)
                total.append(text_label[index] +'--'+pred)
end = time.time()
print(end-st)
with open('preds.txt','w',encoding='utf-8') as f:
    f.write('\n'.join(total))
