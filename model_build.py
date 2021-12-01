import torch
import torch.nn as nn
from kobert_transformers import get_tokenizer
from transformers import ViTFeatureExtractor, VisionEncoderDecoderModel
from processors import TrOCRProcessor, shift_tokens_right
def make_processor():
    tokenizer = get_tokenizer()
    feature_extractor = ViTFeatureExtractor(size=384)
    processor = TrOCRProcessor(feature_extractor, tokenizer)
    return processor

import pickle
import os
if os.path.isfile('processor.pkl') and os.path.isfile('model.pkl'):


    with open('processor.pkl','rb') as f:
        processor = pickle.load(f)
    with open('model.pkl','rb') as f:
        model = pickle.load(f)

else:
    processor = make_processor()
    with open('processor.pkl','wb') as f:
        pickle.dump(processor,f)
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-stage1')
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
    model.decoder.output_projection = None
    model.decoder.output_projection = new_output_projection
    new_embed = nn.Embedding(8002,1024,padding_idx=1); nn.init.uniform_(new_embed.weight, -1.0, 1.0)

    model.decoder.model.decoder.embed_tokens = None
    model.decoder.model.decoder.embed_tokens = new_embed
    # config 수정
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = 8002
    model.config.decoder.vocab_size = 8002

    with open('model.pkl','wb') as f:
        pickle.dump(model,f)




# if os.path.isfile('./new_model.pth'):
#     print('module')
#     model.load_state_dict(torch.load('./new_model.pth'))
