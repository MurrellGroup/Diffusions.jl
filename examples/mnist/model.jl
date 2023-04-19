# The code in this file is derived from the diffusion_mnist example at model-zoo.
# https://github.com/FluxML/model-zoo/tree/master/vision/diffusion_mnist
# The license is MIT: https://github.com/FluxML/model-zoo/blob/master/LICENSE.md

using Diffusions: RandomFourierFeatures
using Flux
using Flux: @functor
using MLUtils: flatten
using Optimisers: Optimisers

@functor RandomFourierFeatures
Optimisers.trainable(::RandomFourierFeatures) = (;)  # no trainable parameters

"""
Create a UNet architecture as a backbone to a diffusion model. \n
# Notes
Images stored in WHCN (width, height, channels, batch) order. \n
In our case, MNIST comes in as (28, 28, 1, batch). \n
# References
paper-  https://arxiv.org/abs/1505.04597
"""
struct UNet
    layers::NamedTuple
end

@functor UNet

"""
User Facing API for UNet architecture.
"""
function UNet(c, channels=[32, 64, 128, 256], embed_dim=256, scale=30.0f0)
    k = 10  # number of classes
    bias = false
    return UNet((
        gaussfourierproj=RandomFourierFeatures(embed_dim, scale),
        labelproj=Embedding(k, embed_dim),
        linear=Dense(embed_dim, embed_dim, swish),
        # Encoding
        conv1=Conv((3, 3), c => channels[1], stride=1, pad=1; bias),
        dense1=Dense(embed_dim, channels[1]),
        gnorm1=GroupNorm(channels[1], 4, swish),
        conv2=Conv((3, 3), channels[1] => channels[2], stride=2, pad=1; bias),
        dense2=Dense(embed_dim, channels[2]),
        gnorm2=GroupNorm(channels[2], 32, swish),
        conv3=Conv((3, 3), channels[2] => channels[3], stride=2, pad=1; bias),
        dense3=Dense(embed_dim, channels[3]),
        gnorm3=GroupNorm(channels[3], 32, swish),
        conv4=Conv((3, 3), channels[3] => channels[4], stride=2, pad=1; bias),
        dense4=Dense(embed_dim, channels[4]),
        gnorm4=GroupNorm(channels[4], 32, swish),
        # Decoding
        tconv4=ConvTranspose((3, 3), channels[4] => channels[3], stride=2, pad=1; bias),
        dense5=Dense(embed_dim, channels[3]),
        tgnorm4=GroupNorm(channels[3], 32, swish),
        tconv3=ConvTranspose((3, 3), channels[3] + channels[3] => channels[2], pad=(1, 0, 1, 0), stride=2; bias),
        dense6=Dense(embed_dim, channels[2]),
        tgnorm3=GroupNorm(channels[2], 32, swish),
        tconv2=ConvTranspose((3, 3), channels[2] + channels[2] => channels[1], pad=(1, 0, 1, 0), stride=2; bias),
        dense7=Dense(embed_dim, channels[1]),
        tgnorm2=GroupNorm(channels[1], 32, swish),
        tconv1=ConvTranspose((3, 3), channels[1] + channels[1] => 1, stride=1, pad=1; bias),
        # Classification
        maxpool=GlobalMaxPool(),
        class=Dense(channels[4] => k, bias=false),
    ))
end

"""
Makes the UNet struct callable and shows an example of a "Functional" API for modeling in Flux. \n
"""
function (unet::UNet)(x, y, t)
    # Embedding
    embed = unet.layers.linear(unet.layers.gaussfourierproj(t) .+ unet.layers.labelproj(y))
    # Encoder
    h1 = unet.layers.conv1(x)
    h1 = h1 .+ expand_dims(unet.layers.dense1(embed), 2)
    h1 = unet.layers.gnorm1(h1)
    h2 = unet.layers.conv2(h1)
    h2 = h2 .+ expand_dims(unet.layers.dense2(embed), 2)
    h2 = unet.layers.gnorm2(h2)
    h3 = unet.layers.conv3(h2)
    h3 = h3 .+ expand_dims(unet.layers.dense3(embed), 2)
    h3 = unet.layers.gnorm3(h3)
    h4 = unet.layers.conv4(h3)
    h4 = h4 .+ expand_dims(unet.layers.dense4(embed), 2)
    h4 = unet.layers.gnorm4(h4)
    # Decoder
    h = unet.layers.tconv4(h4)
    h = h .+ expand_dims(unet.layers.dense5(embed), 2)
    h = unet.layers.tgnorm4(h)
    h = unet.layers.tconv3(cat(h, h3; dims=3))
    h = h .+ expand_dims(unet.layers.dense6(embed), 2)
    h = unet.layers.tgnorm3(h)
    h = unet.layers.tconv2(cat(h, h2; dims=3))
    h = h .+ expand_dims(unet.layers.dense7(embed), 2)
    h = unet.layers.tgnorm2(h)
    h = unet.layers.tconv1(cat(h, h1; dims=3))
    # Classification
    logits = unet.layers.class(flatten(unet.layers.maxpool(h4)))
    return h, logits
end

"""
Helper function that adds `dims` dimensions to the front of a `AbstractVecOrMat`.
Similar in spirit to TensorFlow's `expand_dims` function.
# References:
https://www.tensorflow.org/api_docs/python/tf/expand_dims
"""
expand_dims(x::AbstractVecOrMat, dims::Int=2) = reshape(x, (ntuple(i -> 1, dims)..., size(x)...))
