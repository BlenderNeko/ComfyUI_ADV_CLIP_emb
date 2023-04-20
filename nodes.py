from .adv_encode import advanced_encode

class AdvancedCLIPTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "text": ("STRING", {"multiline": True}),
            "clip": ("CLIP", ),
            "token_normalization": (["none", "mean", "length", "length+mean"],),
            "weight_interpretation": (["comfy", "A1111", "compel", "comfy++"],),
            }}
    
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"

    CATEGORY = "conditioning"

    def encode(self, clip, text, token_normalization, weight_interpretation):
        embeddings_final = advanced_encode(clip, text, token_normalization, weight_interpretation, w_max=1.0)

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