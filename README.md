## More Clip embedding stuff wooh

*give proper explanation and introduction

### node settings
To achieve all of this, the following 3 nodes are introduced:

**token_normalization:** determines how token weights are normalized. Currently supports the following options:
- none: does not alter the weights
- mean: shifts weights such that the mean of all meaningful tokens becomes 1
- length: divides token weight of long words or embeddings between all the tokens, e.g. if an embedding takes up 2 tokens and has a weight of 1.5, the two tokens both receive a weight of 1.25
- length+mean: divides token weight of long words, and then shifts the mean to 1

**attention_method:** determines how up/down weighting should be handled. Currently supports the following options:
- comfy: the default in ComfyUI, CLIP vectors are lerped between the prompt and a completely empty prompt
- A1111: CLip vectors are scaled by their weight
- comfy++: Each word is lerped between the prompt and a prompt where the word is masked off. This is very expensive but the lerp direction might be more accurate/well behaved?

**renorm_method:** determines how the CLIP embedding is scaled back, effects of this appear very minor. Currently supports the following options:
- none: leaves the embeddings alone
- magnitude: scales the embedding to recover the magnitude of the unweighted embedding.
- mean+std: recovers the mean and standard deviation of the unweighted embedding.
- A1111: scales the embedding to recover the mean of the unweighted embedding (no idea, don't ask)