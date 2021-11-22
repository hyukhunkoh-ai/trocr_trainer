import torch
from transformers import (
    TrOCRConfig,
    TrOCRForCausalLM,
    VisionEncoderDecoderModel,
    ViTConfig,
    ViTModel,
)

## trocr 모델 불러오기
def create_rename_keys(encoder_config, decoder_config):
    rename_keys = []
    for i in range(encoder_config.num_hidden_layers):
        # encoder layers: output projection, 2 feedforward neural networks and 2 layernorms
        rename_keys.append(
            (f"encoder.deit.blocks.{i}.norm1.weight", f"encoder.encoder.layer.{i}.layernorm_before.weight")
        )
        rename_keys.append((f"encoder.deit.blocks.{i}.norm1.bias", f"encoder.encoder.layer.{i}.layernorm_before.bias"))
        rename_keys.append(
            (f"encoder.deit.blocks.{i}.attn.proj.weight", f"encoder.encoder.layer.{i}.attention.output.dense.weight")
        )
        rename_keys.append(
            (f"encoder.deit.blocks.{i}.attn.proj.bias", f"encoder.encoder.layer.{i}.attention.output.dense.bias")
        )
        rename_keys.append(
            (f"encoder.deit.blocks.{i}.norm2.weight", f"encoder.encoder.layer.{i}.layernorm_after.weight")
        )
        rename_keys.append((f"encoder.deit.blocks.{i}.norm2.bias", f"encoder.encoder.layer.{i}.layernorm_after.bias"))
        rename_keys.append(
            (f"encoder.deit.blocks.{i}.mlp.fc1.weight", f"encoder.encoder.layer.{i}.intermediate.dense.weight")
        )
        rename_keys.append(
            (f"encoder.deit.blocks.{i}.mlp.fc1.bias", f"encoder.encoder.layer.{i}.intermediate.dense.bias")
        )
        rename_keys.append(
            (f"encoder.deit.blocks.{i}.mlp.fc2.weight", f"encoder.encoder.layer.{i}.output.dense.weight")
        )
        rename_keys.append((f"encoder.deit.blocks.{i}.mlp.fc2.bias", f"encoder.encoder.layer.{i}.output.dense.bias"))

    # cls token, position embeddings and patch embeddings of encoder
    rename_keys.extend(
        [
            ("encoder.deit.cls_token", "encoder.embeddings.cls_token"),
            ("encoder.deit.pos_embed", "encoder.embeddings.position_embeddings"),
            ("encoder.deit.patch_embed.proj.weight", "encoder.embeddings.patch_embeddings.projection.weight"),
            ("encoder.deit.patch_embed.proj.bias", "encoder.embeddings.patch_embeddings.projection.bias"),
            ("encoder.deit.norm.weight", "encoder.layernorm.weight"),
            ("encoder.deit.norm.bias", "encoder.layernorm.bias"),
        ]
    )

    return rename_keys


# we split up the matrix of each encoder layer into queries, keys and values
def read_in_q_k_v(state_dict, encoder_config):
    for i in range(encoder_config.num_hidden_layers):
        # queries, keys and values (only weights, no biases)
        in_proj_weight = state_dict.pop(f"encoder.deit.blocks.{i}.attn.qkv.weight")

        state_dict[f"encoder.encoder.layer.{i}.attention.attention.query.weight"] = in_proj_weight[
                                                                                    : encoder_config.hidden_size, :
                                                                                    ]
        state_dict[f"encoder.encoder.layer.{i}.attention.attention.key.weight"] = in_proj_weight[
                                                                                  encoder_config.hidden_size: encoder_config.hidden_size * 2,
                                                                                  :
                                                                                  ]
        state_dict[f"encoder.encoder.layer.{i}.attention.attention.value.weight"] = in_proj_weight[
                                                                                    -encoder_config.hidden_size:, :
                                                                                    ]


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


def get_original_model(checkpoint_url="trocr-large-handwritten.pt"):
    # size of the architecture
    encoder_config = ViTConfig(image_size=384, qkv_bias=False)
    decoder_config = TrOCRConfig()
    if "base" in checkpoint_url:
        decoder_config.encoder_hidden_size = 768
    elif "large" in checkpoint_url:
        # use ViT-large encoder
        encoder_config.hidden_size = 1024
        encoder_config.intermediate_size = 4096
        encoder_config.num_hidden_layers = 24
        encoder_config.num_attention_heads = 16
        decoder_config.encoder_hidden_size = 1024
    else:
        raise ValueError("Should either find 'base' or 'large' in checkpoint URL")
    #     trmodel = VisionEncoderDecoderModel.from_pretrained(checkpoint_url)
    #     encoder = trmodel.encoder
    #     encoder.training = True
    #     encoder.train()
    #     return trmodel, encoder
    encoder = ViTModel(encoder_config, add_pooling_layer=False)
    decoder = TrOCRForCausalLM(decoder_config)
    model = VisionEncoderDecoderModel(encoder=encoder, decoder=decoder)
    model.train()
    state_dict = torch.load('htrtr.pt')['model']
    rename_keys = create_rename_keys(encoder_config, decoder_config)
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    read_in_q_k_v(state_dict, encoder_config)

    # remove parameters we don't need
    del state_dict["encoder.deit.head.weight"]
    del state_dict["encoder.deit.head.bias"]
    del state_dict["decoder.version"]

    # add prefix to decoder keys
    for key, val in state_dict.copy().items():
        val = state_dict.pop(key)
        if key.startswith("decoder") and "output_projection" not in key:
            state_dict["decoder.model." + key] = val
        else:
            state_dict[key] = val

    # load state dict
    model.load_state_dict(state_dict)
    for param in model.parameters():
        param.requires_grad = False
    #     model.save_pretrained('./trocr_pretrained')
    return model
