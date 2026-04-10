import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----- weight initialization helpers -----

def xavier_init(m):
    s =  np.sqrt( 2. / (m.in_features + m.out_features) )
    m.weight.data.normal_(0, s)


def he_init(m):
    """Initialize weights using He (Kaiming) initialization.

    Scales the standard deviation by sqrt(2 / fan_in), which keeps the
    variance of activations roughly constant across layers when using ReLU
    activations.  Xavier initialization (fan_in + fan_out) is derived for
    linear/tanh activations and underestimates the scale needed for ReLU,
    causing gradients to vanish in deep networks.  He init corrects for
    this and is the standard choice for ReLU-based architectures.

    Args:
        m: an nn.Linear module whose weight tensor will be modified in-place.
    """
    s =  np.sqrt( 2. / m.in_features )
    m.weight.data.normal_(0, s)


def normal_init(m, mean=0., std=0.01):
    """Initialize weights by sampling from N(mean, std).

    A conservative alternative to He/Xavier when a small, fixed scale is
    preferred (e.g. output heads where large initial logits are undesirable).

    Args:
        m:    an nn.Linear module.
        mean: mean of the normal distribution (default 0).
        std:  standard deviation (default 0.01, intentionally small).
    """
    m.weight.data.normal_(mean, std)


# ----- basic building blocks -----

class CReLU(nn.Module):
    """Concatenated ReLU: applies ReLU to both x and -x, then concatenates.

    This doubles the output dimension but preserves both positive and negative
    activation information, which can improve gradient flow.  Note: the
    current forward implementation has a bug (missing dim argument to
    torch.cat) and is unused in the codebase.
    """
    def __init__(self):
        super(CReLU, self).__init__()

    def forward(self, x):
        return torch.cat( F.relu(x), F.relu(-x), 1 )


class NonLinear(nn.Module):
    """A linear layer with an optional trailing activation function.

    This is a thin wrapper around nn.Linear that makes activation an explicit
    constructor parameter, keeping model definitions declarative.  When
    activation=None the module is a plain affine transformation — useful for
    output heads (e.g. mu/log-var projections) that must not be squashed.

    Args:
        input_size:  number of input features.
        output_size: number of output features.
        bias:        whether to include a bias term (default True).
        activation:  an instantiated nn.Module (e.g. nn.ReLU()) or None.

    Returns:
        Tensor of shape (batch, output_size).
    """
    def __init__(self, input_size, output_size, bias=True, activation=None):
        super(NonLinear, self).__init__()

        self.activation = activation
        self.linear = nn.Linear(int(input_size), int(output_size), bias=bias)

    def forward(self, x):
        h = self.linear(x)          # (batch, output_size)
        if self.activation is not None:
            h = self.activation( h )

        return h


# ----- gated dense / convolutional layers -----

class GatedDense(nn.Module):
    """Gated dense layer: h(x) * sigmoid(g(x)).

    The gating mechanism multiplies a content branch h(x) element-wise by a
    soft gate g(x) in [0, 1].  Each output unit learns to independently
    suppress or pass its own signal, giving the network a form of
    multiplicative control that pure additive layers cannot express.  This is
    analogous to the update-gate in a GRU and has been shown to help training
    stability in deep generative models.

    When no_attention=True the gate branch is removed entirely and the layer
    falls back to a standard dense layer with a ReLU activation.  This lets
    the same constructor be used in ablations or architectures that do not
    need multiplicative gating.

    Args:
        input_size:   number of input features.
        output_size:  number of output features.
        activation:   optional activation applied to h(x) before gating.
                      When no_attention=False this is typically None (the gate
                      alone provides nonlinearity).
        no_attention: if True, disable gating and use ReLU instead.

    Returns:
        Tensor of shape (batch, output_size).
    """
    def __init__(self, input_size, output_size, activation=None, no_attention=False):
        super(GatedDense, self).__init__()

        self.activation = activation
        self.no_attention = no_attention
        self.sigmoid = nn.Sigmoid()
        self.h = nn.Linear(input_size, output_size)   # content branch
        if no_attention is False:
            self.g = nn.Linear(input_size, output_size)  # gate branch
        else:
            # No gate — fall back to a plain nonlinear layer with ReLU.
            self.activation = torch.nn.ReLU()

    def forward(self, x):
        h = self.h(x)                                  # (batch, output_size)
        if self.activation is not None:
            h = self.activation( self.h( x ) )         # (batch, output_size)
        try:
            if self.no_attention is False:
                g = self.sigmoid(self.g(x))            # (batch, output_size), values in (0, 1)
                return h * g                           # element-wise gating
            else:
                return h
        except:
            g = self.sigmoid(self.g(x))
            return h * g


class GatedConv2d(nn.Module):
    """Gated 2-D convolution: activation(h(x)) * sigmoid(g(x)).

    The convolutional analogue of GatedDense.  Two parallel convolutions share
    the same spatial configuration — one produces content, the other a
    per-spatial-location soft gate — and their outputs are multiplied
    element-wise.  This mechanism is used in PixelCNN-style models to control
    information flow through the feature maps.

    When no_attention=True the gate branch is removed and ELU is used as a
    plain activation instead.

    Args:
        input_channels:  C_in.
        output_channels: C_out for both the content and gate convolutions.
        kernel_size:     spatial kernel size.
        stride:          convolution stride.
        padding:         zero-padding on each spatial side.
        dilation:        dilation factor (default 1).
        activation:      optional activation applied to the content branch
                         before gating (e.g. nn.ELU()).
        no_attention:    if True, disable gating (see note above).

    Returns:
        Tensor of shape (batch, C_out, H_out, W_out).
    """
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding, dilation=1, activation=None,
                 no_attention=False):
        super(GatedConv2d, self).__init__()
        self.no_attention = no_attention

        self.activation = activation
        self.sigmoid = nn.Sigmoid()

        self.h = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, dilation)  # content
        if no_attention is False:
            self.g = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, dilation)  # gate
        else:
            self.activation = torch.nn.ELU()

    def forward(self, x):
        if self.activation is None:
            h = self.h(x)                              # (batch, C_out, H_out, W_out)
        else:
            h = self.activation( self.h( x ) )         # (batch, C_out, H_out, W_out)

        if self.no_attention is False:
            g = self.sigmoid( self.g( x ) )            # (batch, C_out, H_out, W_out), values in (0, 1)
            return h * g                               # element-wise spatial gating
        else:
            return h                                   # plain activation, no gating


class Conv2d(nn.Module):
    """Thin wrapper around nn.Conv2d with an optional activation.

    Mirrors the NonLinear design but for 2-D convolutions, keeping model
    definitions consistent.  When activation=None the output is the raw
    convolution result (e.g. for the final logit layer).

    Args:
        input_channels:  C_in.
        output_channels: C_out.
        kernel_size:     spatial kernel size.
        stride:          convolution stride.
        padding:         zero-padding per side.
        dilation:        dilation factor (default 1).
        activation:      optional instantiated activation module or None.
        bias:            whether to include a bias term (default True).

    Returns:
        Tensor of shape (batch, C_out, H_out, W_out).
    """
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding, dilation=1, activation=None, bias=True):
        super(Conv2d, self).__init__()

        self.activation = activation
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, dilation, bias=bias)

    def forward(self, x):
        h = self.conv(x)                               # (batch, C_out, H_out, W_out)
        if self.activation is None:
            out = h
        else:
            out = self.activation(h)

        return out


class GatedConvTranspose2d(nn.Module):
    """Gated transposed 2-D convolution: activation(h(x)) * sigmoid(g(x)).

    The transpose (upsampling) analogue of GatedConv2d.  Two parallel transposed
    convolutions share the same spatial configuration — one produces content, the
    other a per-spatial-location soft gate — and their outputs are multiplied
    element-wise.  Useful in decoder / generator networks where spatial
    resolution needs to increase while retaining gated information flow.

    When no_attention=True the gate branch is removed and ELU is used as a
    plain activation.

    Args:
        input_channels:  C_in.
        output_channels: C_out for both the content and gate transposed convolutions.
        kernel_size:     spatial kernel size.
        stride:          convolution stride (controls upsampling factor).
        padding:         zero-padding on each spatial side.
        output_padding:  additional size added to one side of the output (default 0).
        dilation:        dilation factor (default 1).
        activation:      optional activation applied to the content branch
                         before gating (e.g. nn.ELU()).
        no_attention:    if True, disable gating and use ELU activation instead.

    Returns:
        Tensor of shape (batch, C_out, H_out, W_out).
    """
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding,
                 output_padding=0, dilation=1, activation=None, no_attention=False):
        super(GatedConvTranspose2d, self).__init__()
        self.no_attention = no_attention

        self.activation = activation
        self.sigmoid = nn.Sigmoid()

        self.h = nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, padding,
                                    output_padding, dilation=dilation)  # content
        if no_attention is False:
            self.g = nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, padding,
                                        output_padding, dilation=dilation)  # gate
        else:
            self.activation = torch.nn.ELU()

    def forward(self, x):
        if self.activation is None:
            h = self.h(x)                              # (batch, C_out, H_out, W_out)
        else:
            h = self.activation(self.h(x))             # (batch, C_out, H_out, W_out)

        if self.no_attention is False:
            g = self.sigmoid(self.g(x))                # (batch, C_out, H_out, W_out), values in (0, 1)
            return h * g                               # (batch, C_out, H_out, W_out) — element-wise spatial gating
        else:
            return h                                   # (batch, C_out, H_out, W_out)


# ----- autoregressive / masked convolutions (PixelCNN) -----

class MaskedConv2d(nn.Conv2d):
    """Causal convolutional layer for autoregressive image models (PixelCNN).

    Zeros out weights that would look at future (right/below) pixel positions,
    enforcing the autoregressive factorisation p(x) = prod_i p(x_i | x_{<i}).

    Two mask types:
      'A' — masks the center pixel as well, used for the very first layer so
            that no output depends on the pixel being predicted.
      'B' — keeps the center pixel, used for subsequent layers where the
            center is a transformed representation of the past, not the raw
            pixel value.

    The mask is stored as a buffer (not a parameter) so it moves with the
    module to the correct device but is not updated by the optimizer.

    Args:
        mask_type: 'A' or 'B' (see above).
        *args, **kwargs: forwarded to nn.Conv2d.
    """
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        # Zero out the center pixel for type 'A', keep it for type 'B'.
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, kH // 2 + 1:] = 0             # zero all rows below center

    def forward(self, x):
        # Apply the mask in-place before each forward pass so that
        # gradient updates never reintroduce masked-out connections.
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)


# ----- PixelSNAIL building blocks -----
# Copyright (c) Xi Chen
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Borrowed from https://github.com/neocxi/pixelsnail-public and ported it to PyTorch

from math import sqrt
from functools import partial, lru_cache

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def wn_linear(in_dim, out_dim):
    """Weight-normalized linear layer.

    Weight normalization decouples the magnitude of the weight vector from its
    direction, which can stabilise and accelerate training by reducing
    sensitivity to the learning rate.
    """
    return nn.utils.weight_norm(nn.Linear(in_dim, out_dim))


class WNConv2d(nn.Module):
    """Weight-normalized 2-D convolution with an optional activation.

    Used throughout the PixelSNAIL stack instead of plain Conv2d to get the
    training-stability benefits of weight normalization without having to call
    nn.utils.weight_norm at every use site.

    Args:
        in_channel:   C_in.
        out_channel:  C_out.
        kernel_size:  int or (H, W) tuple.
        stride:       convolution stride (default 1).
        padding:      zero-padding per side (default 0).
        bias:         include bias (default True).
        activation:   optional instantiated activation module or None.

    Returns:
        Tensor of shape (batch, C_out, H_out, W_out).
    """
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        stride=1,
        padding=0,
        bias=True,
        activation=None,
    ):
        super().__init__()

        self.conv = nn.utils.weight_norm(
            nn.Conv2d(
                in_channel,
                out_channel,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            )
        )

        self.out_channel = out_channel

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size, kernel_size]

        self.kernel_size = kernel_size

        self.activation = activation

    def forward(self, input):
        out = self.conv(input)                         # (batch, C_out, H_out, W_out)

        if self.activation is not None:
            out = self.activation(out)

        return out


# ----- causal shift helpers -----

def shift_down(input, size=1):
    """Shift the feature map down by `size` rows (pad top, crop bottom).

    Used to implement the vertical stack in PixelCNN++: after a causal
    'down' convolution, shifting ensures that position (i, j) can only attend
    to rows strictly above row i.
    """
    return F.pad(input, [0, 0, size, 0])[:, :, : input.shape[2], :]


def shift_right(input, size=1):
    """Shift the feature map right by `size` columns (pad left, crop right).

    Used to implement the horizontal stack: after a causal 'downright'
    convolution, shifting ensures that position (i, j) only attends to pixels
    strictly to its left on the same row.
    """
    return F.pad(input, [size, 0, 0, 0])[:, :, :, : input.shape[3]]


class CausalConv2d(nn.Module):
    """Causal 2-D convolution for the PixelSNAIL vertical/horizontal stacks.

    Enforces causality through asymmetric zero-padding rather than weight
    masking, which is more efficient for large kernels.

    Three padding modes:
      'downright' — pads only top and left; the receptive field covers the
                    upper-left triangle (used for the vertical stack).
      'down'      — pads symmetrically left/right and only on top; the
                    receptive field covers current and upper rows.
      'causal'    — like 'down' but additionally zeros the right half of the
                    bottom kernel row at runtime, preventing the current
                    position from seeing to its right on the same row.

    Args:
        in_channel:   C_in.
        out_channel:  C_out.
        kernel_size:  int or [H, W].
        stride:       default 1.
        padding:      'downright', 'down', or 'causal'.
        activation:   passed through to the inner WNConv2d.
    """
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        stride=1,
        padding='downright',
        activation=None,
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * 2

        self.kernel_size = kernel_size

        if padding == 'downright':
            # Pad only top and left so the kernel cannot see below or to the right.
            pad = [kernel_size[1] - 1, 0, kernel_size[0] - 1, 0]

        elif padding == 'down' or padding == 'causal':
            pad = kernel_size[1] // 2
            # Symmetric horizontal padding + top-only vertical padding.
            pad = [pad, pad, kernel_size[0] - 1, 0]

        self.causal = 0
        if padding == 'causal':
            # Remember how many right-side columns to zero out at forward time.
            self.causal = kernel_size[1] // 2

        self.pad = nn.ZeroPad2d(pad)

        self.conv = WNConv2d(
            in_channel,
            out_channel,
            kernel_size,
            stride=stride,
            padding=0,           # all padding already handled by self.pad
            activation=activation,
        )

    def forward(self, input):
        out = self.pad(input)

        if self.causal > 0:
            # Zero the right half of the bottom kernel row so position (i, j)
            # cannot attend to (i, j+1 .. j+causal) on the same row.
            self.conv.conv.weight_v.data[:, :, -1, self.causal :].zero_()

        out = self.conv(out)

        return out


class GatedResBlock(nn.Module):
    """Residual block with a Gated Linear Unit (GLU) output gate.

    Architecture:
      activation -> conv1 -> [+ aux] -> activation -> dropout -> conv2
      -> [+ condition] -> GLU -> + input (skip connection)

    The GLU halves the channel dimension: conv2 outputs 2*in_channel features,
    which GLU splits into (content, gate) and returns content * sigmoid(gate).
    This gives the block data-dependent output suppression while the residual
    connection preserves gradient flow.

    Optional conditioning (class labels, etc.) is injected additively before
    the gate via a 1x1 WNConv2d projection, enabling class-conditional
    generation without modifying the core residual path.

    Args:
        in_channel:         number of input (and output) channels.
        channel:            number of internal channels for conv1.
        kernel_size:        spatial kernel size.
        conv:               conv type — 'wnconv2d', 'causal_downright', or 'causal'.
        activation:         activation class (not instance), default nn.ELU.
        dropout:            dropout probability after the second activation.
        auxiliary_channel:  if > 0, add an aux feature map via a 1x1 conv.
        condition_dim:      if > 0, add a conditioning signal via a 1x1 conv.
    """
    def __init__(
        self,
        in_channel,
        channel,
        kernel_size,
        conv='wnconv2d',
        activation=nn.ELU,
        dropout=0.1,
        auxiliary_channel=0,
        condition_dim=0,
    ):
        super().__init__()

        if conv == 'wnconv2d':
            conv_module = partial(WNConv2d, padding=kernel_size // 2)

        elif conv == 'causal_downright':
            conv_module = partial(CausalConv2d, padding='downright')

        elif conv == 'causal':
            conv_module = partial(CausalConv2d, padding='causal')

        self.activation = activation()
        self.conv1 = conv_module(in_channel, channel, kernel_size)

        if auxiliary_channel > 0:
            # 1x1 conv to project aux features to the same channel count as conv1 output.
            self.aux_conv = WNConv2d(auxiliary_channel, channel, 1)

        self.dropout = nn.Dropout(dropout)

        # Output 2*in_channel so GLU can split into content + gate halves.
        self.conv2 = conv_module(channel, in_channel * 2, kernel_size)

        if condition_dim > 0:
            # self.condition = nn.Linear(condition_dim, in_channel * 2, bias=False)
            # 1x1 conv instead of Linear so the conditioning map can be spatial.
            self.condition = WNConv2d(condition_dim, in_channel * 2, 1, bias=False)

        # GLU along dim=1 (channel axis): splits in_channel*2 → in_channel.
        self.gate = nn.GLU(1)

    def forward(self, input, aux_input=None, condition=None):
        out = self.conv1(self.activation(input))       # (batch, channel, H, W)

        if aux_input is not None:
            out = out + self.aux_conv(self.activation(aux_input))  # (batch, channel, H, W)

        out = self.activation(out)
        out = self.dropout(out)
        out = self.conv2(out)                          # (batch, in_channel*2, H, W)

        if condition is not None:
            condition = self.condition(condition)      # (batch, in_channel*2, H, W)
            out += condition
            # out = out + condition.view(condition.shape[0], 1, 1, condition.shape[1])

        out = self.gate(out)                           # (batch, in_channel, H, W) — GLU halves channels
        out += input                                   # residual connection

        return out


@lru_cache(maxsize=64)
def causal_mask(size):
    """Compute a lower-triangular causal attention mask for sequence length `size`.

    Returns two tensors:
      mask        — (1, size, size) uint8, lower-triangular (1 = attend, 0 = mask).
      start_mask  — (size, 1) float32, zeros the first position so it produces
                    no attention output (the autoregressive start token has no
                    context to attend to).

    Results are cached by size so repeated calls with the same image resolution
    do not reallocate tensors.
    """
    shape = [size, size]
    mask = np.triu(np.ones(shape), k=1).astype(np.uint8).T   # lower-triangular
    start_mask = np.ones(size).astype(np.float32)
    start_mask[0] = 0

    return (
        torch.from_numpy(mask).unsqueeze(0),           # (1, size, size)
        torch.from_numpy(start_mask).unsqueeze(1),     # (size, 1)
    )


class CausalAttention(nn.Module):
    """Multi-head causal self-attention over flattened 2-D feature maps.

    Positions are flattened in raster scan order (row-major), and the causal
    mask ensures that position i can only attend to positions 0..i-1.  This
    gives the model a global receptive field while preserving the
    autoregressive ordering required for exact likelihood computation.

    Weight normalization is applied to all linear projections for training
    stability (consistent with the rest of the PixelSNAIL stack).

    Args:
        query_channel: channel dimension of the query input.
        key_channel:   channel dimension of the key/value input.
        channel:       total projected dimension (split across n_head heads).
        n_head:        number of attention heads (default 8).
        dropout:       attention weight dropout probability (default 0.1).
    """
    def __init__(self, query_channel, key_channel, channel, n_head=8, dropout=0.1):
        super().__init__()

        self.query = wn_linear(query_channel, channel)
        self.key = wn_linear(key_channel, channel)
        self.value = wn_linear(key_channel, channel)

        self.dim_head = channel // n_head   # dimension per head
        self.n_head = n_head

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key):
        batch, _, height, width = key.shape

        def reshape(input):
            # Split channel dim into (n_head, dim_head) and move heads to dim 1.
            return input.view(batch, -1, self.n_head, self.dim_head).transpose(1, 2)
            # -> (batch, n_head, seq_len, dim_head)

        # Flatten spatial dims: (batch, C, H, W) -> (batch, H*W, C)
        query_flat = query.view(batch, query.shape[1], -1).transpose(1, 2)
        key_flat = key.view(batch, key.shape[1], -1).transpose(1, 2)

        query = reshape(self.query(query_flat))                   # (batch, n_head, H*W, dim_head)
        key   = reshape(self.key(key_flat)).transpose(2, 3)       # (batch, n_head, dim_head, H*W)
        value = reshape(self.value(key_flat))                     # (batch, n_head, H*W, dim_head)

        # Scaled dot-product attention.
        attn = torch.matmul(query, key) / sqrt(self.dim_head)    # (batch, n_head, H*W, H*W)
        mask, start_mask = causal_mask(height * width)
        mask = mask.type_as(query)
        start_mask = start_mask.type_as(query)
        # Fill future positions with a large negative value before softmax.
        attn = attn.masked_fill(mask == 0, -1e4)
        attn = torch.softmax(attn, 3) * start_mask               # (batch, n_head, H*W, H*W)
        attn = self.dropout(attn)

        out = attn @ value                                        # (batch, n_head, H*W, dim_head)
        # Merge heads and restore spatial layout.
        out = out.transpose(1, 2).reshape(
            batch, height, width, self.dim_head * self.n_head
        )                                                         # (batch, H, W, channel)
        out = out.permute(0, 3, 1, 2)                            # (batch, channel, H, W)

        return out


class PixelBlock(nn.Module):
    """One stage of PixelSNAIL: a stack of causal residual blocks + optional attention.

    Each block applies n_res_block causal GatedResBlocks followed by (optionally)
    a CausalAttention layer that gives the model a global receptive field.
    Without attention the block falls back to a simple 1x1 projection.

    The `background` tensor (normalised pixel coordinates) is concatenated to
    the query and key before attention so the model knows where in the image
    each feature vector comes from.

    Args:
        in_channel:    number of input/output channels.
        channel:       number of internal channels for the residual blocks.
        kernel_size:   spatial kernel for the causal convolutions.
        n_res_block:   number of GatedResBlocks before attention.
        attention:     whether to include the CausalAttention sub-module.
        dropout:       dropout probability.
        condition_dim: if > 0, pass conditioning to each GatedResBlock.
    """
    def __init__(
        self,
        in_channel,
        channel,
        kernel_size,
        n_res_block,
        attention=True,
        dropout=0.1,
        condition_dim=0,
    ):
        super().__init__()

        resblocks = []
        for i in range(n_res_block):
            resblocks.append(
                GatedResBlock(
                    in_channel,
                    channel,
                    kernel_size,
                    conv='causal',
                    dropout=dropout,
                    condition_dim=condition_dim,
                )
            )

        self.resblocks = nn.ModuleList(resblocks)

        self.attention = attention

        if attention:
            # Key: concatenate the pre-block input, post-resblock output, and
            # background coordinates so the key encodes both context and position.
            self.key_resblock = GatedResBlock(
                in_channel * 2 + 2, in_channel, 1, dropout=dropout
            )
            # Query: only the post-resblock output + background.
            self.query_resblock = GatedResBlock(
                in_channel + 2, in_channel, 1, dropout=dropout
            )

            self.causal_attention = CausalAttention(
                in_channel + 2, in_channel * 2 + 2, in_channel // 2, dropout=dropout
            )

            # Merge attention output (aux) into the main path.
            self.out_resblock = GatedResBlock(
                in_channel,
                in_channel,
                1,
                auxiliary_channel=in_channel // 2,
                dropout=dropout,
            )

        else:
            # No attention: project concatenated (out, background) back to in_channel.
            self.out = WNConv2d(in_channel + 2, in_channel, 1)

    def forward(self, input, background, condition=None):
        out = input

        for resblock in self.resblocks:
            out = resblock(out, condition=condition)   # (batch, in_channel, H, W)

        if self.attention:
            # Build keys from both the raw input and the transformed output for
            # richer context, then queries from the transformed output only.
            key_cat = torch.cat([input, out, background], 1)   # (batch, in_channel*2+2, H, W)
            key = self.key_resblock(key_cat)                   # (batch, in_channel, H, W)
            query_cat = torch.cat([out, background], 1)        # (batch, in_channel+2, H, W)
            query = self.query_resblock(query_cat)             # (batch, in_channel, H, W)
            attn_out = self.causal_attention(query, key)       # (batch, in_channel//2, H, W)
            out = self.out_resblock(out, attn_out)             # (batch, in_channel, H, W)

        else:
            bg_cat = torch.cat([out, background], 1)           # (batch, in_channel+2, H, W)
            out = self.out(bg_cat)                             # (batch, in_channel, H, W)

        return out


class CondResNet(nn.Module):
    """Conditioning encoder: maps a class map through a WNConv + GatedResBlock stack.

    Produces a spatial feature map used as the conditioning signal in
    PixelSNAIL.  A one-hot class volume is encoded into a multi-scale
    representation that can be injected into each PixelBlock.

    Args:
        in_channel:   number of input channels (typically n_class for one-hot input).
        channel:      output channel count.
        kernel_size:  spatial kernel size for the initial convolution and resblocks.
        n_res_block:  number of GatedResBlocks after the initial convolution.
    """
    def __init__(self, in_channel, channel, kernel_size, n_res_block):
        super().__init__()

        blocks = [WNConv2d(in_channel, channel, kernel_size, padding=kernel_size // 2)]

        for i in range(n_res_block):
            blocks.append(GatedResBlock(channel, channel, kernel_size))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)                      # (batch, channel, H, W)


# ----- top-level autoregressive model -----

class PixelSNAIL(nn.Module):
    """PixelSNAIL: autoregressive image model combining causal convolutions and attention.

    Factorises the image distribution as p(x) = prod_i p(x_i | x_{<i}) using
    a dual-stack (horizontal + vertical) causal convolutional architecture
    augmented with multi-head causal self-attention for long-range dependencies.

    Two learned position-coordinate channels (normalised to [-0.5, 0.5]) are
    concatenated to feature maps before attention so the model can resolve
    spatial position without positional embeddings.

    Args:
        shape:              (height, width) of the input image.
        n_class:            number of discrete pixel values (e.g. 256).
        channel:            base channel count for the main feature maps.
        kernel_size:        spatial kernel size (forced to odd internally).
        n_block:            number of PixelBlocks.
        n_res_block:        GatedResBlocks per PixelBlock.
        res_channel:        internal channels inside each GatedResBlock.
        attention:          whether each PixelBlock uses CausalAttention.
        dropout:            dropout probability throughout.
        n_cond_res_block:   number of CondResNet blocks (0 = no conditioning).
        cond_res_channel:   channel count for the conditioning encoder output.
        cond_res_kernel:    kernel size for CondResNet.
        n_out_res_block:    extra GatedResBlocks before the final logit projection.
    """
    def __init__(
        self,
        shape,
        n_class,
        channel,
        kernel_size,
        n_block,
        n_res_block,
        res_channel,
        attention=True,
        dropout=0.1,
        n_cond_res_block=0,
        cond_res_channel=0,
        cond_res_kernel=3,
        n_out_res_block=0,
    ):
        super().__init__()

        height, width = shape

        self.n_class = n_class

        # Ensure the kernel is odd so the center pixel is well-defined.
        if kernel_size % 2 == 0:
            kernel = kernel_size + 1

        else:
            kernel = kernel_size

        # Horizontal stack: wide kernel, causal in the 'down' direction.
        self.horizontal = CausalConv2d(
            3, channel, [kernel // 2, kernel], padding='down'
        )
        # Vertical stack: tall kernel, causal in the 'downright' direction.
        self.vertical = CausalConv2d(
            3, channel, [(kernel + 1) // 2, kernel // 2], padding='downright'
        )

        # Pre-computed normalised (x, y) coordinate grids registered as a
        # buffer so they are moved to the correct device with the model.
        coord_x = (torch.arange(height).float() - height / 2) / height
        coord_x = coord_x.view(1, 1, height, 1).expand(1, 1, height, width)  # (1, 1, H, W)
        coord_y = (torch.arange(width).float() - width / 2) / width
        coord_y = coord_y.view(1, 1, 1, width).expand(1, 1, height, width)   # (1, 1, H, W)
        self.register_buffer('background', torch.cat([coord_x, coord_y], 1)) # (1, 2, H, W)

        self.blocks = nn.ModuleList()

        for i in range(n_block):
            self.blocks.append(
                PixelBlock(
                    channel,
                    res_channel,
                    kernel_size,
                    n_res_block,
                    attention=attention,
                    dropout=dropout,
                    condition_dim=cond_res_channel,
                )
            )

        if n_cond_res_block > 0:
            self.cond_resnet = CondResNet(
                n_class, cond_res_channel, cond_res_kernel, n_cond_res_block
            )

        out = []

        for i in range(n_out_res_block):
            out.append(GatedResBlock(channel, res_channel, 1))

        # Final 1x1 conv projects to n_class logits per spatial position.
        out.extend([nn.ELU(inplace=True), WNConv2d(channel, n_class, 1)])

        self.out = nn.Sequential(*out)

    def forward(self, input, condition=None, cache=None):
        """
        Args:
            input:     (batch, C, H, W) — raw pixel tensor.
            condition: (batch,) int tensor of class labels, or None.
            cache:     dict for caching the processed conditioning map across
                       multiple autoregressive steps (avoids re-encoding).

        Returns:
            Logits of shape (batch, n_class, H, W).
        """
        if cache is None:
            cache = {}
        batch, _, height, width = input.shape
        #input = (
        #    F.one_hot(input, self.n_class).permute(0, 3, 1, 2).type_as(self.background)
        #)
        # Shift after causal conv so position (i,j) cannot see itself.
        horizontal = shift_down(self.horizontal(input))    # (batch, channel, H, W)
        vertical = shift_right(self.vertical(input))       # (batch, channel, H, W)
        # Sum the two stacks; this is the standard PixelCNN++ dual-stack merge.
        out = horizontal + vertical                        # (batch, channel, H, W)

        # Expand pre-computed coordinate background to the batch size.
        background = self.background[:, :, :height, :].expand(batch, 2, height, width)

        if condition is not None:
            if 'condition' in cache:
                # Re-use the encoded conditioning map from a previous step.
                condition = cache['condition']
                condition = condition[:, :, :height, :]

            else:
                condition = (
                    F.one_hot(condition, self.n_class)
                    .permute(0, 3, 1, 2)
                    .type_as(self.background)
                )                                          # (batch, n_class, H, W)
                condition = self.cond_resnet(condition)    # (batch, cond_res_channel, H, W)
                # Upsample condition to match input spatial resolution.
                condition = F.interpolate(condition, scale_factor=2)
                cache['condition'] = condition.detach().clone()
                condition = condition[:, :, :height, :]

        for block in self.blocks:
            out = block(out, background, condition=condition)  # (batch, channel, H, W)

        out = self.out(out)                                # (batch, n_class, H, W)

        return out
