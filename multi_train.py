# from torch._C import T
from torch.autograd.grad_mode import no_grad
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from kobert_transformers import get_tokenizer
from transformers import ViTFeatureExtractor, VisionEncoderDecoderModel, AdamW
from tqdm.auto import tqdm
from processors import TrOCRProcessor, shift_tokens_right
from custom_dataset import CustomDataset
from transforms import *
from torch import nn
from lr_schedulers import get_scheduler
from model_build import model, processor
import torch.multiprocessing as mp
import torch
import torch.nn as nn
import torch.distributed as dist
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('Device:', device)
print('Current cuda device:', torch.cuda.current_device())
print('Count of using GPUs:', torch.cuda.device_count())

from skimage import io
import cv2

def loadImage(img_file):
    img = io.imread(img_file)           # RGB order
    if img.shape[0] == 2: img = img[0]
    if len(img.shape) == 2 : img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:   img = img[:,:,:3]
    img = np.array(img)

#학습

# model.to(device)
import datetime
num_epochs = 20
optimizer = AdamW(model.parameters(), lr=2e-03)
when_to_load = {'0':True,'1':True,'2':True,'3':True}

def embeding_train(gpu,args):
    
    
    ############################################################
    #  Initialize the process and join up with the other processes
    rank = gpu              
    dist.init_process_group(                                   
    	backend='nccl',   # fastest backend among others( Gloo, NCCL 및 MPI의 세 가지 백엔드)                                   
   		init_method='tcp://127.0.0.1:5678',  # tells the process group where to look for some settings                       
    	world_size=args.gpus,                              
    	rank=rank                                               
    )                                                          
    ############################################################
    torch.cuda.set_device(gpu) 
  
    batch_size = 30
    optimizer = AdamW(model.parameters(), lr=1e-3)
    lr_scheduler = get_scheduler(optimizer,warmup=5)
    model.cuda(gpu)
    ###############################################################
    # Wrap the model to each gpu
    dist_model = nn.parallel.DistributedDataParallel(model,
                                                device_ids=[gpu])
    if os.path.isfile('./finetune_model_52.pth'):
        dist.barrier()
        if when_to_load[str(gpu)]:
            print('load successfully')
            map_location = {'cuda:%d' % 0: 'cuda:%d' % gpu}    
            dist_model.module.load_state_dict(torch.load('./finetune_model_52.pth', map_location=map_location))
            when_to_load[str(gpu)] = False

    print('continue')
    ################################################################
    # assign different slice of data per the process 
    root_data_dir = './data/'
    df = CustomDataset(None,processor=processor, data_dir=root_data_dir).df
    train_df, test_df = train_test_split(df, test_size=0.01)
    # # we reset the indices to start from zero
    # train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)
    
    train_dataset = CustomDataset(df,processor=processor, data_dir=root_data_dir,tfms=train_tfms) ###############path 수정
    eval_dataset = CustomDataset(test_df,processor=processor, data_dir=root_data_dir)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
    	train_dataset,
    	num_replicas=args.gpus,
    	rank=rank
    )
    ################################################################
    
    train_loader = torch.utils.data.DataLoader(
    	dataset=train_dataset,
       batch_size=batch_size,
    ##############################
       shuffle=False,            #
    ##############################
       num_workers=4,
       pin_memory=True,
    #############################
      sampler=train_sampler)    # 
    #############################
    eval_loader = torch.utils.data.DataLoader(
    	dataset=eval_dataset,
       batch_size=batch_size,
    ##############################
       shuffle=False,            #
    ##############################
       num_workers=4,
       pin_memory=True)
    #############################
    start = datetime.datetime.now()
    total_step = len(train_loader)
    best_trloss = 1000000
    best_valloss = 1000000
    best_epoch = 0
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        total_trloss = 0
        total_valloss = 0    
        for i, batch in enumerate(train_loader):
            img_batch = batch['pixel_values'].cuda(gpu)
            labels_batch = batch['labels'].cuda(gpu)
            outputs = dist_model(pixel_values=img_batch, labels=labels_batch)
            preds = outputs.logits.argmax(-1)
            
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            if gpu == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                    epoch + 1, 
                    num_epochs, 
                    i + 1, 
                    total_step,
                    loss.item())
                )
            total_trloss += loss.item()
        with torch.no_grad():
            for i, batch in enumerate(eval_loader):
                img_batch = batch['pixel_values'].cuda(gpu)
                labels_batch = batch['labels'].cuda(gpu)
                outputs = dist_model(pixel_values=img_batch, labels=labels_batch)
                preds = outputs.logits.argmax(-1)
                loss = outputs.loss
                total_valloss += loss.item()
                if gpu == 0:
                    print(processor.tokenizer.decode(preds[-1]))
        if gpu == 0:
            print(total_trloss,total_valloss)
            print(best_epoch,best_valloss)
            if (total_trloss < best_trloss) and total_valloss < best_valloss:
                print(epoch)
                print('------------save & loss below---------------')
                # best_lr = lr_scheduler.get_lr()
                best_epoch = epoch
                best_trloss = total_trloss
                best_valloss = total_valloss
                # print(best_lr)
                torch.save(dist_model.module.state_dict(), f'embed_model_{best_epoch}.pth')

    if gpu == 0:
        print("Training complete in: " + str(datetime.datetime.now() - start))
        torch.save(dist_model.module.state_dict(), 'embedding_model.pth')

def finetune_train(gpu,args):
    
    ############################################################
    #  Initialize the process and join up with the other processes
    rank = gpu              
    dist.init_process_group(                                   
    	backend='nccl',   # fastest backend among others( Gloo, NCCL 및 MPI의 세 가지 백엔드)                                   
   		init_method='tcp://127.0.0.1:5678',  # tells the process group where to look for some settings                       
    	world_size=args.gpus,                              
    	rank=rank                                               
    )                                                          
    ############################################################
    torch.cuda.set_device(gpu) 
  
    batch_size = 64
    optimizer = AdamW(model.parameters(), lr=1e-3)
    lr_scheduler = get_scheduler(optimizer,num_epochs*0.05)
    model.cuda(gpu)
    ###############################################################
    # Wrap the model to each gpu
    dist_model = nn.parallel.DistributedDataParallel(model,
                                                device_ids=[gpu])
    pretrained_name = f'./embedding_model.pth'
    if os.path.isfile(pretrained_name):
        dist.barrier()
        if when_to_load[str(gpu)]:
            print('load successfully')
            map_location = {'cuda:%d' % 0: 'cuda:%d' % gpu}    
            dist_model.module.load_state_dict(torch.load(pretrained_name, map_location=map_location))
            when_to_load[str(gpu)] = False

    print('continue')
    dist_model들 간의 통신 연결 - hook 등록
    ###############################################################

    ################################################################
    # assign different slice of data per the process 
    root_data_dir = './data/'
    df = CustomDataset(None,processor=processor, data_dir=root_data_dir).df
    train_df, test_df = train_test_split(df, test_size=0.01)
    # # we reset the indices to start from zero
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)
#     test_df = CustomDataset(None,processor=processor, data_dir='./data/siyeon/').df
    train_dataset = CustomDataset(train_df,processor=processor, data_dir=root_data_dir,tfms=train_tfms) ###############path 수정
    eval_dataset = CustomDataset(test_df,processor=processor, data_dir=root_data_dir)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
    	train_dataset,
    	num_replicas=args.gpus,
    	rank=rank
    )
    ################################################################
    
    train_loader = torch.utils.data.DataLoader(
    	dataset=train_dataset,
       batch_size=batch_size,
    ##############################
       shuffle=False,            #
    ##############################
       num_workers=4,
       pin_memory=True,
    #############################
      sampler=train_sampler)    # 
    #############################
    eval_loader = torch.utils.data.DataLoader(
    	dataset=eval_dataset,
       batch_size=batch_size,
    ##############################
       shuffle=False,            #
    ##############################
       num_workers=4,
       pin_memory=True)
    #############################
    start = datetime.datetime.now()
    total_step = len(train_loader)
    best_trloss = 1000000
    best_valloss = 10000.0
    best_epoch = 0
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        total_trloss = 0
        total_valloss = 0    
        for i, batch in enumerate(train_loader):
            img_batch = batch['pixel_values'].cuda(gpu)
            labels_batch = batch['labels'].cuda(gpu)
            outputs = dist_model(pixel_values=img_batch, labels=labels_batch)
            preds = outputs.logits.argmax(-1)
            
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            if gpu == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                    epoch + 1, 
                    num_epochs, 
                    i + 1, 
                    total_step,
                    loss.item())
                )
                if loss.item() > 2.0:
                     print('labels : ', batch['text'][-1])
            total_trloss += loss.item()
        with torch.no_grad():
            for i, batch in enumerate(eval_loader):
                img_batch = batch['pixel_values'].cuda(gpu)
                labels_batch = batch['labels'].cuda(gpu)
                outputs = dist_model(pixel_values=img_batch, labels=labels_batch)
                preds = outputs.logits.argmax(-1)
                loss = outputs.loss
                total_valloss += loss.item()
                if gpu == 0:
                    print(processor.tokenizer.batch_decode(preds,skip_special_tokens=True))
                    print(batch['text'])
        
                   
        if gpu == 0:
            print(total_trloss,total_valloss)
            print(best_epoch,best_valloss)
            if (total_trloss < best_trloss) and total_valloss < best_valloss:
                print(epoch)
                print('------------save & loss below---------------')
                # best_lr = lr_scheduler.get_lr()
                best_epoch = epoch
                best_trloss = total_trloss
                best_valloss = total_valloss
                # print(best_lr)
                torch.save(dist_model.module.state_dict(), f'finetune_{best_epoch}.pth')
            # if total_trloss/(22*513) < 5.0:
            #     raise ValueError("early stopping")

    if gpu == 0:
        print("Training complete in: " + str(datetime.datetime.now() - start))

def single_train():
    # lr_scheduler = get_scheduler(optimizer)
    pretrained_name = f'./last_finetune_{0}.pth'
    if os.path.isfile(pretrained_name):
        model.load_state_dict(torch.load(pretrained_name, map_location='cpu'))
    root_data_dir = './data/siyeon/'
    num_epochs = 100
    optimizer = AdamW(model.parameters(), lr=5e-03)
    df = CustomDataset(None,processor=processor, data_dir=root_data_dir).df
    train_dataset = CustomDataset(df,processor=processor, data_dir=root_data_dir,tfms=train_tfms) ###############path 수정
    train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=32,shuffle=True)
    progress_bar = tqdm(range(num_epochs * len(train_dataloader)))
    model.to(device)
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            print('one_batch_done')
            img_batch = batch['pixel_values'].to(device)
            labels_batch = batch['labels'].to(device)
            outputs = model(pixel_values=img_batch, labels=labels_batch)
            preds = outputs.logits.argmax(-1)
            # outputs = model(pixel_values=img_batch, labels=labels_batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            # lr_scheduler.step()
            optimizer.zero_grad()
            # print(model.encoder._modules['encoder']._modules['layer']._modules['0']._modules['output']._modules['dense'].weight)
            progress_bar.update(1)
            print(labels_batch[-1])
            print(processor.tokenizer.decode(labels_batch[-1]))
            print(processor.tokenizer.decode(preds[-1]))

    torch.save(model.state_dict(), 'new_model.pth')
        

if __name__ == '__main__':
    

    import argparse
    parser = argparse.ArgumentParser(description='how to train?')
    parser.add_argument('--mode', help='an integer for the accumulator')
    parser.add_argument('--gpus', default=3,type=int,help='how many gpu use?')
    parser.add_argument('--devices', help='what gpu use?',default='1,2,3')
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES']= args.devices
    if args.mode == 'embed':
        mp.spawn(embeding_train, nprocs=args.gpus,args=(args,))
    elif args.mode == 'finetune':
        for param in model.decoder.model.decoder.layers.parameters():
            param.requires_grad = True
        for param in model.decoder.output_projection.parameters():
            param.requires_grad = True
        for param in model.decoder.model.decoder.embed_tokens.parameters():
            param.requires_grad = True
        for param in model.decoder.model.decoder.embed_positions.parameters():
            param.requires_grad = True
        for param in model.encoder.encoder.layer.parameters():
            param.requires_grad = True
        for param in model.encoder.embeddings.patch_embeddings.parameters():
            param.requires_grad = True
        for param in model.encoder.layernorm.parameters():
            param.requires_grad = True
        for param in model.encoder.pooler.parameters():
            param.requires_grad = True
        mp.spawn(finetune_train, nprocs=args.gpus,args=(args,))
    elif args.mode =='single':
        for param in model.decoder.model.decoder.layers.parameters():
            param.requires_grad = True
        for param in model.decoder.output_projection.parameters():
            param.requires_grad = True
        for param in model.decoder.model.decoder.embed_tokens.parameters():
            param.requires_grad = True
        for param in model.decoder.model.decoder.embed_positions.parameters():
            param.requires_grad = True
        for param in model.encoder.encoder.layer.parameters():
            param.requires_grad = True
        for param in model.encoder.embeddings.patch_embeddings.parameters():
            param.requires_grad = True
        for param in model.encoder.layernorm.parameters():
            param.requires_grad = True
        for param in model.encoder.pooler.parameters():
            param.requires_grad = True
        single_train()



    # single_train()
    # from collections import OrderedDict
    # new_state_dict = OrderedDict()
    # for k,v in state_dict.items():
    #     name = k[7:]
    #     new_state_dict[name] = v

    
