import torch
import torch.nn as nn
import open_clip
import os
from collections import OrderedDict

class Text2pc(nn.Module):
    def __init__(self, text_file, clip_model):
        super(Text2pc, self).__init__()
        with open(text_file, 'r') as f:
             class_names = f.read().splitlines()
        # prompts = [f"a 3d model of {name}" for name in class_names]

        for param in clip_model.parameters():
                param.requires_grad = False

        self.text_encoder = TextEncoder(clip_model)

        self.prompt_learner = VLPromptLearner(class_names, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts


    def forward(self, pc_features):
        # 计算 text features (N_cls, 512)
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        self.text_feat = self.text_encoder(prompts, tokenized_prompts) # (N_cls, 512)

        print(self.text_feat.requires.grad)

        # 计算 pc features 与 text features 的相似度 (B, N_cls)
        similarity_matrix = torch.matmul(pc_features, self.text_feat.T)

        print(similarity_matrix.grad)

        # 应用 Gumbel Softmax，得到新的 (B, N_cls) 矩阵
        softmaxed_matrix = nn.functional.gumbel_softmax(similarity_matrix, tau=1, hard=False)

        print(softmaxed_matrix.grad)

        # 新的矩阵与 text features 相乘，得到 (B, 512)
        combined_features = torch.matmul(softmaxed_matrix, self.text_feat)

        print(combined_features.grad)

        # 最终与 image features 相加得到 (B, 512) 输出
        outputs = combined_features + pc_features
        print(outputs.grad)

        return outputs, self.text_feat

class VLPromptLearner(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        dtype = clip_model.text_projection.dtype
        ctx_init = "a 3d model of a"
        # No prompting
        ctx_init = ctx_init.replace("_", " ")
        prompt_prefix = ctx_init
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([open_clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
        self.register_buffer("complete_text_embeddings", embedding)
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor

    def forward(self):
        prompts = self.complete_text_embeddings

        return prompts

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection

        print(clip_model.text_projection.shape)
        self.dtype = clip_model.text_projection.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x









