# Advanced CLIP Text Encode

This repo contains 4 nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) that allows for more control over the way prompt weighting should be interpreted.

---

### BNK_CLIPTextEncodeAdvanced node settings
To achieve this, a CLIP Text Encode (Advanced) node is introduced with the following 2 settings:

#### token_normalization:
determines how token weights are normalized. Currently supports the following options:
- **none**: does not alter the weights.
- **mean**: shifts weights such that the mean of all meaningful tokens becomes 1.
- **length**: divides token weight of long words or embeddings between all the tokens. It does so in a manner that the magnitude of the weight change remains constant between different lengths of tokens. E.g. if a word is expressed as 3 tokens and it has a weight of 1.5 all tokens get a weight of around 1.29 because sqrt(3 * pow(0.35, 2)) = 0.5.
- **length+mean**: divides token weight of long words, and then shifts the mean to 1.

#### weight_interpretation:
Determines how up/down weighting should be handled. Currently supports the following options:
- **comfy**: the default in ComfyUI, CLIP vectors are lerped between the prompt and a completely empty prompt.
- **A1111**: CLip vectors are scaled by their weight
- **compel**: Interprets weights similar to [compel](https://github.com/damian0815/compel). Compel up-weights the same as comfy, but mixes masked embeddings to accomplish down-weighting (more on this later).
- **comfy++**: When up-weighting, each word is lerped between the prompt and a prompt where the word is masked off. Additionally uses compel style down-weighting.
- **down_weight**: rescales weights such that the maximum weight is one. This means that you will only ever be down-weighting. Uses compel style down-weighting.

<details>
<summary>
Intuition behind weight interpretation methods
</summary>

### up weighting

the diagram below visualizes the 3 different way in which the 3 methods to transform the clip embeddings to achieve up-weighting

![visual explanation of attention methods](https://github.com/BlenderNeko/ComfyUI_ADV_CLIP_emb/blob/master/visual.png)

As can be seen, in A1111 we use weights to travel on the line between the zero vector and the vector corresponding to the token embedding. This can be seen as adjusting the magnitude of the embedding which both makes our final embedding point more in the direction the thing we are up weighting (or away when down weighting) and creates stronger activations out of SD because of the bigger numbers.

Comfy also creates a direction starting from a single point but instead uses the vector embedding corresponding to a completely empty prompt. we are now traveling on a line that approximates the epitome of a certain thing. Despite the magnitude of the vector not growing as fast as in A1111 this is actually quite effective and can result in SD quite aggressively chasing concepts that are up-weighted.

Comfy++ does not start from a single point but instead travels between the presence and absence of a concept in the prompt. Despite the idea being similar to that of comfy it is a lot less aggressive.

#### visual comparison of the different methods

Below a short clip of the prompt `cinematic wide shot of the ocean, beach, (palmtrees:1.0), at sunset, milkyway`, where the weight of palmtree slowly increasses from 1.0 to 2.0 in 20 steps. (made using [silicon29](https://huggingface.co/Xynon/SD-Silicon) in SD 1.5)

https://user-images.githubusercontent.com/126974546/232336840-e9076b7c-3799-4335-baaa-992a6b8cad8a.mp4

### down-weighting

One of the issues with using the above methods for down-weighting is that the embedding vectors associated with a token do not just contain "information" about that token, but actually pull in a lot of context about the entire prompt. Most of the information they contain seemingly is about that specific token, which is why theses various up-weighting interpretations work, but that given token permeates throughout the entire CLIP embedding. In the example prompt above we can down-weight `palmtrees` all the way to .1 in comfy or A1111, but because the presence of the tokens that represent palmtrees affects the entire embedding, we still get to see a lot of palmtrees in our outputs. suppose we have the prompt `(pears:.2) and (apples:.5) in a bowl`. Compel does the following to accomplish down-weighting: it creates embeddings 
- `A` = `pears and apples in a bowl`, 
- `B` = `_ and apples in a bowl`
-  `C` = `_ and _ in a bowl`

which it then mixes into a final embedding `0.2 * A + 0.3 * B + 0.5 * C`. This way we truly only have 0.2 of the influence of pears in our entire embedding, and 0.5 of apples.

</details>

---

### Mix Clip Embeddings node (Depricated)

The functionality of this node can now be found in the core ComfyUI nodes.

---

## SDXL support

To support SDXL the following settings and nodes are provided. Note that the CLIP Text Encode (Advanced) node also works just fine for SDXL :

---

### BNK_CLIPTextEncodeSDXLAdvanced

The CLIP Text Encode SDXL (Advanced) node provides the same settings as its non SDXL version. In addition it also comes with 2 text fields to send different texts to the two CLIP models. and with the following setting:

- **balance**: tradeoff between the CLIP and openCLIP models. At 0.0 the embedding only contains the CLIP model output and the contribution of the openCLIP model is zeroed out. At 1.0 the embedding only contains the openCLIP model and the CLIP model is entirely zeroed out.

This node mainly exists for experimentation.

---

### BNK_AddCLIPSDXLParams

the Add CLIP SDXL Params node adds the following SDXL parameters to a conditioning:

- **width**: width of the image crop.
- **height**: height of the image crop.
- **crop_w**: left pixel of the crop.
- **crop_h**: top pixel of the crop.
- **target_width**: width of the original image.
- **target_height**: height of the original image.

----

### BNK_AddCLIPSDXLRParams

the Add CLIP SDXL Refiner Params node adds the following refiner parameters to a conditioning:

- **width**: width of the image.
- **height**: height of the image.
- **ascore**: aesthetic score of the image.