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


# class VLPromptLearner(nn.Module):   #加prompt
#     def __init__(self, classnames, clip_model):
#         super().__init__()
#         dtype=clip_model.text_projection.dtype
#         n_cls = len(classnames)
#         n_ctx = 10
#         ctx_dim = clip_model.ln_final.weight.shape[0]
#         # random initialization
#         ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
#         nn.init.normal_(ctx_vectors, std=0.02)
#         prompt_prefix = " ".join(["X"] * n_ctx)
#         # prompt_s = " ".join(["X"] * (n_ctx/2))
#
#         self.ctx = nn.Parameter(ctx_vectors)
#
#         classnames = [name.replace("_", " ") for name in classnames]
#         # prompts = [prompt_prefix + " " + name + " " + prompt_s + "." for name in classnames]
#         prompts = [prompt_prefix + " " + name + "." for name in classnames]
#         tokenized_prompts = torch.cat([open_clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
#         with torch.no_grad():
#             embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
#         self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
#         self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS
#         self.n_cls = n_cls
#         self.tokenized_prompts = tokenized_prompts  # torch.Tensor
#
#     def construct_prompts(self, ctx, prefix, suffix, label=None):
#         # dim0 is either batch_size (during training) or n_cls (during testing)
#         # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
#         # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
#         # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)
#
#         if label is not None:
#             prefix = prefix[label]
#             suffix = suffix[label]
#
#         prompts = torch.cat(
#             [
#                 prefix,  # (dim0, 1, dim)
#                 ctx,  # (dim0, n_ctx, dim)
#                 suffix,  # (dim0, *, dim)
#             ],
#             dim=1,
#         )
#
#         return prompts
#
#     def forward(self):
#         ctx = self.ctx
#         if ctx.dim() == 2:
#             ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
#
#         prefix = self.token_prefix
#         suffix = self.token_suffix
#         prompts = self.construct_prompts(ctx, prefix, suffix)
#
#         return prompts

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









