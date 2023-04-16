import numpy as np
import torch
from tqdm.auto import trange
import pickle

class AdvancedCLIPTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "text": ("STRING", {"multiline": True}),
            "clip": ("CLIP", ),
            "token_normalization": (["none", "mean", "length", "length+mean"],),
            "attention_method": (["comfy", "A1111", "comfy++"],),
            "renorm_method": (["none", "magnitude", "mean+std", "A1111"],),

            }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"

    CATEGORY = "conditioning"

    def mask_word_id(self,tokens, word_ids, target_id, mask_token):
        new_tokens = [[mask_token if wid == target_id else t for t, wid in zip(x,y)] for x,y in zip(tokens, word_ids)]
        mask = np.array(word_ids) == target_id
        return (new_tokens, mask)

    def norm_mag(self, w, n):
        d = w - 1
        return  1 + np.sign(d) * np.sqrt(np.abs(d)**2 / n)
        

    def encode(self, clip, text, token_normalization, attention_method, renorm_method):

        tokenized = clip.tokenize(text, return_word_ids=True)
        tokens = [[t for t,_,_ in x] for x in tokenized]
        weights = [[w for _,w,_ in x] for x in tokenized]
        word_ids = [[wid for _,_,wid in x] for x in tokenized]

        #weight normalization

        #distribute down/up weights over word lengths
        if token_normalization.startswith("length"):
            sums = dict(zip(*np.unique(word_ids, return_counts=True)))
            sums[0] = 1
            weights = [[self.norm_mag(w, sums[id]) for w, id in zip(x, y)] for x, y in zip(weights, word_ids)]
        
        #make mean of word tokens 1
        if token_normalization.endswith("mean"):
            delta = 1 - np.mean([w for x, y in zip(weights, word_ids) for  w, id in zip(x,y) if id != 0])
            weights = [[w if id == 0 else w+delta for w, id in zip(x, y)] for x, y in zip(weights, word_ids)]

        #attention

        #calc unweighted embeddings
        if attention_method != 'comfy' or renorm_method != 'none':
            unweighted_tokens = [[(t,1.0) for t, _,_ in x] for x in tokenized]
            base_emb = clip.encode_from_tokens(unweighted_tokens)

        #use comfy attention
        if attention_method == "comfy":
            weighted_tokens = [[(t,w) for t, w in zip(x, y)] for x, y in zip(tokens, weights)]
            weighted_emb = clip.encode_from_tokens(weighted_tokens)
        else:
            weight_tensor = torch.tensor(weights, dtype=base_emb.dtype, device=base_emb.device)
            weight_tensor = weight_tensor.reshape(1,-1,1).expand(base_emb.shape)

        #use A1111 attention
        if attention_method == "A1111":
            weighted_emb = base_emb * weight_tensor

        #calc attention by masking per word
        if attention_method == "comfy++":
            word_count = np.max(word_ids)
            embs = [torch.zeros_like(base_emb)]
            wids, inds = np.unique(np.array(word_ids).reshape(-1), return_index=True)
            weight_dict = dict(zip(wids ,np.array(weights).reshape(-1)[inds]))
            
            for i in trange(word_count):
                if weight_dict[i+1] == 1.0:
                    continue
                masked_tokens, mask = self.mask_word_id(tokens, word_ids, i+1, clip.tokenizer.end_token)
                masked_tokens = [[(t,1.0) for t in x] for x in masked_tokens]
                
                mask = torch.tensor(mask, dtype=base_emb.dtype, device=base_emb.device)
                mask = mask.reshape(1,-1,1).expand(base_emb.shape)
                
                emb = clip.encode_from_tokens(masked_tokens)
                emb = (base_emb - emb) * mask
                embs.append(emb)

            embs = torch.stack(embs).sum(axis=0)
            weighted_emb = base_emb + ((weight_tensor - 1) * embs)

        #embedding renormalization
        if renorm_method == "none":
            embeddings_final = weighted_emb

        if renorm_method == "magnitude":
            norm_base = torch.linalg.norm(base_emb)
            norm_weighted = torch.linalg.norm(weighted_emb)
            embeddings_final = (norm_base / norm_weighted) * weighted_emb

        if renorm_method == "mean+std":
            fixed_std = (base_emb.std() / weighted_emb.std()) * (weighted_emb - weighted_emb.mean())
            embeddings_final = fixed_std + (base_emb.mean() - fixed_std.mean())

        if renorm_method == "A1111":
            embeddings_final = (base_emb.mean() / weighted_emb.mean()) * weighted_emb

        return ([[embeddings_final, {}]], )
    
class MixCLIPEmbeddings:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "cond1":("CONDITIONING",),
            "cond2": ("CONDITIONING",),
            "mix": ("FLOAT", {"default": .5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "mix_embs"

    CATEGORY = "conditioning"
        
    def mix_embs(self, cond1, cond2, mix):
        conditioning = []
        for c1, c2 in zip (cond1, cond2):
            if c1[0].shape == c2[0].shape:
                c_mixed = c1[0] * (1-mix) + c2[0] * mix
            else:
                print('warning, CLIP embedding size mismatch, ignoring mix')
                c_mixed = c1[0]
            conditioning.append([c_mixed, c1[1].copy()])
        return (conditioning, )
    
NODE_CLASS_MAPPINGS = {
    "AdvancedCLIPTextEncode": AdvancedCLIPTextEncode,
    "MixCLIPEmbeddings": MixCLIPEmbeddings,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AdvancedCLIPTextEncode" : "CLIP Text Encode (Advanced)",
    "MixCLIPEmbeddings" : "Mix CLIP Embeddings"
}