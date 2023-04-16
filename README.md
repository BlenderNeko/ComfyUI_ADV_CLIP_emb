## Advanced CLIP Text Encode

This repo contains 2 nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) that allows for more control over the way prompt weighting should be interpreted. And let's you mix different embeddings.

### Advanced CLIP Text Encode node settings
To achieve this, an Advanced Clip Text Encode node is introduced with the following 3 settings:

#### token_normalization:
determines how token weights are normalized. Currently supports the following options:
- **none**: does not alter the weights
- **mean**: shifts weights such that the mean of all meaningful tokens becomes 1
- **length**: divides token weight of long words or embeddings between all the tokens. It does so in a manner that the magnitude of the weight change remains constant between different lengths of tokens. E.g. if a word is expressed as 3 tokens and it has a weight of 1.5 all tokens get a weight of around 1.29 because sqrt(3 * pow(0.35, 2)) = 0.5
- **length+mean**: divides token weight of long words, and then shifts the mean to 1

#### attention_method:
determines how up/down weighting should be handled. Currently supports the following options:
- **comfy**: the default in ComfyUI, CLIP vectors are lerped between the prompt and a completely empty prompt
- **A1111**: CLip vectors are scaled by their weight
- **comfy++**: Each word is lerped between the prompt and a prompt where the word is masked off. When a lot of up/down weighting is happening in the prompt this becomes quite expensive but the lerp direction might be more accurate/well behaved? This is similar to how [compel](https://github.com/damian0815/compel) does down-weighting, but here we use it for both down and up.

#### renorm_method:
determines how the CLIP embedding is scaled back, effects of this appear very minor. Currently supports the following options:
- **none**: leaves the embeddings alone
- **magnitude**: scales the embedding to recover the magnitude of the unweighted embedding.
- **mean+std**: recovers the mean and standard deviation of the unweighted embedding.
- **A1111**: scales the embedding to recover the mean of the unweighted embedding (if you feel the need to use this, the magnitude option does roughly the same thing but is more mathematically grounded).

### Intuition behind attention methods

the diagram below visualizes the way in which the 3 methods transform the clip embeddings to achieve the weighting

![visual explanation of attention methods](https://github.com/BlenderNeko/ComfyUI_ADV_CLIP_emb/blob/master/visual.png)

As can be seen, in A1111 we use weights to travel on the line between the zero vector and the vector corresponding to the token embedding. This can be seen as adjusting the magnitude of the embedding which both makes our final embedding point more in the direction the thing we are up weighting (or away when down weighting) and creates stronger activations out of SD because of the bigger numbers.

Comfy also creates a direction starting from a single point but instead uses the vector embedding corresponding to a completely empty prompt. we are now traveling on a line that approximates the epitome of a certain thing. Despite the magnitude of the vector not growing as fast as in A1111 this is actually quite effective and can result in SD quite aggressively chasing concepts that are up-weighted.

Comfy++ does not start from a single point but instead travels between the presence and absence of a concept in the prompt. Despite the idea being similar to that of comfy it is a lot less aggressive, and behaves more like A111.

### Mix Clip Embeddings node

The Mix Clip Embeddings node takes two conditionings as input and mixes them according to a mix factor. When 0.0 the output will fully be the embeddings in the top slot, and when 1.0 it will fully be those of the embeddings in the bottom slot. Only the area conditioning of the top slot will pass through the mix node.

This node lets you interpolate between the different approaches given by the Advanced CLIP Text Encode node or even mix between completely different encodings. The only constraint here is that both encodings must be of the same size, if one of the prompts fits inside 77 tokens but the other prompt requires 154, the node will ignore the mixing factor and pass through the top embedding.

## TODOs:
- [ ] smart area conditioning pass through logic