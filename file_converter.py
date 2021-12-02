###################
import numpy
np_im = numpy.array(image)
np_array = np_im[24:,:]
pils=Image.fromarray(np_array)
#################



####################
import os
paths = './data'
with open(f'{paths}/labels.txt', 'r',encoding='utf-8') as f:
    data = f.read().split('\n')
# data = [i for i in data if len(i.split('.jpg')) != 2]
data = data[:-1]

res =  []
adds = 0 # typical index add, for example, images exist 300.jpg then start by 301
for idx,img in enumerate(data):
    name,label = img.split('.jpg ')
    res.append(f'{paths}/images/{idx+adds}.jpg ' + label)
    if os.path.isfile(f'{paths}/images/{name}.jpg'):
        os.rename(f'{paths}/images/{name}.jpg', f'{paths}/images/{idx+adds}.jpg')
####################
with open('new_labels.txt','w',encoding='utf-8') as f:
    f.write('\n'.join(res))
