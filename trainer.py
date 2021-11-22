from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from kobert_transformers import get_tokenizer
from transformers import ViTFeatureExtractor, VisionEncoderDecoderModel, AdamW
from tqdm.auto import tqdm
from processors import TrOCRProcessor
from custom_dataset import CustomDataset
from transforms import *
from torch import nn
from lr_schedulers import get_scheduler
#######################################################path 수정
# tokenizer = KoBertTokenizer('./kotrocr/refine_vocab.txt','./kotrocr/sp_model.pkl')
def make_processor():
    tokenizer = get_tokenizer()
    feature_extractor = ViTFeatureExtractor(size=384)
    processor = TrOCRProcessor(feature_extractor, tokenizer)
    return processor



processor = make_processor()
root_data_dir = './kotrocr/data/'
## dataset만들기 -- custom dataset 추가
# img, labels -> [[img1, label1(labelin,labelout)],[img2, label2(labelin,labelout)]
df = CustomDataset(None,processor=processor, data_dir=root_data_dir).df
train_df, test_df = train_test_split(df, test_size=0.2)
# we reset the indices to start from zero
train_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)
train_dataset = CustomDataset(train_df,processor=processor, data_dir=root_data_dir,tfms=train_tfms) ###############path 수정
eval_dataset = CustomDataset(test_df,processor=processor, data_dir=root_data_dir)
## dataloader mapping
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8)
eval_dataloader = DataLoader(eval_dataset, batch_size=8)




import torch
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#모델불러오기
model = VisionEncoderDecoderModel.from_pretrained('./kotrocr/original_trocr.pth',local_files_only=True)
#train모드로 바꾸고 freeze하기
model.train()
model.decoder.train()
model.encoder.train()
for param in model.decoder.parameters():
    param.requires_grad = False
for param in model.parameters():
    param.requires_grad = False


# tokenzier layer와 outputlayer 갈아끼기
new_output_projection = nn.Linear(1024,8002);torch.nn.init.xavier_uniform_(new_output_projection.weight)
model.decoder.output_projection = new_output_projection
new_embed = nn.Embedding(8002,1024,padding_idx=1); nn.init.uniform_(new_embed.weight, -1.0, 1.0)
model.decoder.model.decoder.embed_tokens = new_embed
# config 수정
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size


#학습

model.to(device)


num_epochs = 1
optimizer = AdamW(model.parameters(), lr=2e-05)
lr_scheduler = get_scheduler(optimizer)
progress_bar = tqdm(range(num_epochs * len(train_dataloader)))
for epoch in range(num_epochs):
    for batch in train_dataloader:
        print('one_batch_done')
        img_batch = batch['pixel_values'].to(device)
        labels_batch = batch['labels'].to(device)
        outputs = model(pixel_values=img_batch, labels=labels_batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        print(model.encoder._modules['encoder']._modules['layer']._modules['0']._modules['output']._modules['dense'].weight)
        progress_bar.update(1)

        
#평가
# from datasets import load_metric

# metric = load_metric("accuracy")
# model.eval()
# for batch in eval_dataloader:
#     batch = {k: v.to(device) for k, v in batch.items()}
#     with torch.no_grad():
#         outputs = model(**batch)

#     logits = outputs.logits
#     predictions = torch.argmax(logits, dim=-1)
#     metric.add_batch(predictions=predictions, references=batch["labels"])

# metric.compute()
