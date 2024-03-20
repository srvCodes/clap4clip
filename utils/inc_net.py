import copy
import logging
import torch
from torch import nn
from backbone.linears import SimpleLinear, SplitCosineLinear, CosineLinear
import timm

def get_backbone(args, pretrained=False):
    name = args["backbone_type"].lower()
    # SimpleCIL or SimpleCIL w/ Finetune
    if name == "pretrained_vit_b16_224" or name == "vit_base_patch16_224":
        model = timm.create_model("vit_base_patch16_224",pretrained=True, num_classes=0)
        model.out_dim = 768
        return model.eval()
    elif name == "pretrained_vit_b16_224_in21k" or name == "vit_base_patch16_224_in21k":
        model = timm.create_model("vit_base_patch16_224_in21k",pretrained=True, num_classes=0)
        model.out_dim = 768
        return model.eval()
    # L2P
    elif '_l2p' in name:
        if args["model_name"] == "l2p":
            model = timm.create_model(
                args["backbone_type"],
                pretrained=args["pretrained"],
                num_classes=args["nb_classes"],
                drop_rate=args["drop"],
                drop_path_rate=args["drop_path"],
                drop_block_rate=None,
                prompt_length=args["length"],
                embedding_key=args["embedding_key"],
                prompt_init=args["prompt_key_init"],
                prompt_pool=args["prompt_pool"],
                prompt_key=args["prompt_key"],
                pool_size=args["size"],
                top_k=args["top_k"],
                batchwise_prompt=args["batchwise_prompt"],
                prompt_key_init=args["prompt_key_init"],
                head_type=args["head_type"],
                use_prompt_mask=args["use_prompt_mask"],
            )
            return model
        else:
            raise NotImplementedError("Inconsistent model name and model type")
    # dualprompt
    elif '_dualprompt' in name:
        if args["model_name"] == "dualprompt":
            model = timm.create_model(
                args["backbone_type"],
                pretrained=args["pretrained"],
                num_classes=args["nb_classes"],
                drop_rate=args["drop"],
                drop_path_rate=args["drop_path"],
                drop_block_rate=None,
                prompt_length=args["length"],
                embedding_key=args["embedding_key"],
                prompt_init=args["prompt_key_init"],
                prompt_pool=args["prompt_pool"],
                prompt_key=args["prompt_key"],
                pool_size=args["size"],
                top_k=args["top_k"],
                batchwise_prompt=args["batchwise_prompt"],
                prompt_key_init=args["prompt_key_init"],
                head_type=args["head_type"],
                use_prompt_mask=args["use_prompt_mask"],
                use_g_prompt=args["use_g_prompt"],
                g_prompt_length=args["g_prompt_length"],
                g_prompt_layer_idx=args["g_prompt_layer_idx"],
                use_prefix_tune_for_g_prompt=args["use_prefix_tune_for_g_prompt"],
                use_e_prompt=args["use_e_prompt"],
                e_prompt_layer_idx=args["e_prompt_layer_idx"],
                use_prefix_tune_for_e_prompt=args["use_prefix_tune_for_e_prompt"],
                same_key_value=args["same_key_value"],
            )
            return model
        else:
            raise NotImplementedError("Inconsistent model name and model type")