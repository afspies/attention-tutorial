""" Jax implementation of the Slot-Attention Module / Auto-encoder by Locatello et al.
    Paper: https://arxiv.org/abs/2006.15055
    Git: https://github.com/google-research/google-research/tree/master/slot_attention
"""
import jax
import jax.numpy as jnp
import haiku as hk

class SlotAttentionAE(hk.Module):
    def __init__(self, C, attention_fn, position_enc_fn, name='SlotAttentionAE'):
        super().__init__(name=name)
        self.encoder = CNNEncoder(C, position_enc_fn, name=None)
        self.slots = SlotAttentionModule(C, attention_fn, name=None)
        self.decoder = CNNDecoder(C, position_enc_fn, name=None)

    def __call__(self, x, iterations=3, get_attn=False):
        x = self.encoder(x)
        x = self.slots(x, iterations, get_attn)
        if get_attn:
            x, attn = x
        x = self.decoder(x)
        return (x, attn) if get_attn else x

class CNNDecoder(hk.Module):
    def __init__(self, C, position_enc_fn, name='SlotAttDecoder'):
        super().__init__(name=name)
        self.C = C
        self.num_slots = C['slots']
        channels, kernels, strides = C['decoder_cnn_channels'], C['decoder_cnn_kernels'], C['decoder_cnn_strides']

        deconv_layers = [
           	hk.Conv2DTranspose(channels[0], kernels[0], stride=strides[0], padding='SAME'), jax.nn.relu,
            hk.Conv2DTranspose(channels[1], kernels[1], stride=strides[1], padding='SAME'), jax.nn.relu,
            hk.Conv2DTranspose(channels[2], kernels[2], stride=strides[2], padding='SAME'), jax.nn.relu,
            hk.Conv2DTranspose(4, kernels[3], stride=strides[3]),
        ]

        self.deconvolutions = hk.Sequential(deconv_layers)

        self.pos_embed = SoftPositionEmbed(C['slot_size'], C['spatial_broadcast_dims'], position_enc_fn)

    def __call__(self, x, return_masks=False):
        x = self.tile_grid(x)
        x = self.pos_embed(x)
        x = self.deconvolutions(x)
        x, alphas = self.decollapse_and_split(x)
        alphas = jax.nn.softmax(alphas, axis=1) # Softmax across slots
        x = jnp.sum(x * alphas, axis=1, keepdims=False) # Sum across slots
        if return_masks:
            return (x, alphas)
        else:
            return x

    def decollapse_and_split(self, x):        
        # Decollapse batches and split alpha from color channels
        x = jnp.reshape(x, (x.shape[0]//self.num_slots, self.num_slots, *x.shape[1:])) # Decollapse batches from slots
        x, alphas = jnp.array_split(x, [x.shape[-1]-1], -1)
        return x, alphas

    def tile_grid(self, x):
        # takes slots (batch, k, d) and returns (batch*k, w, h, d)
        # i.e. collapse batches (for computation/layer applicability?) and copy slot information wxh times, wtf?
        # maybe this general representational mapping format is sensible - grid cells and conceptual spaces eichenbaum hmmm
        x = jnp.reshape(x, (x.shape[0]*x.shape[1], 1, 1, x.shape[-1]))
        return jnp.tile(x, [1,*self.C['spatial_broadcast_dims'],1])
    

class SlotAttentionModule(hk.Module):
    """Slot Attention Module - Iteratively perform dot product attention over inputs
        Inputs are (32*32, hidden_dim) and slots are (num_slots, slot_dim)
    """
    def __init__(self, C, attention_fn, name='SlotAttention'):
        super().__init__(name=name)
        he_init = hk.initializers.VarianceScaling(scale=2.0)

        self.num_slots = C['slots']
        self.slot_size = C['slot_size'] 
        self.attn_eps = C['attention_eps']
        self.mlp_hidden_size = C['mlp_hidden_size']
        # Learnable mu and sigma (no covar) (dim slot_dim) to initilialize slots

        # Learnable linear transforms for K,Q,V Attention - Use Glorot init here?
        self.k = hk.Linear(self.slot_size, w_init=he_init, with_bias=False)
        self.q = hk.Linear(self.slot_size, w_init=he_init, with_bias=False)
        self.v = hk.Linear(self.slot_size, w_init=he_init, with_bias=False)

        # Layer norm for slots after and before attention (GRU Omitted)
        self.layer_norm_in = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        self.layer_norm_1 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        self.layer_norm_2 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)

        # Slot update function is learned by GRU - #hidden_states = slot_dim
        self.mlp = hk.Sequential([
            hk.Linear(self.mlp_hidden_size, w_init=he_init), jax.nn.relu,# MLP + Residual Connection improves output
            hk.Linear(self.slot_size, w_init=he_init)
        ])

        self.attention_fn = attention_fn

    def __call__(self, x, T:int, get_attn=False):
        x = self.layer_norm_in(x)
        # Glorot uniform initialization for the sampling distrib. params
        mu = hk.get_parameter("mu", [self.slot_size], init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform'))
        logstd = hk.get_parameter("logstd", [self.slot_size], init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform'))
        # Initialize slots from common distribution - Affine transform of norm. dist
        slots = mu + jnp.exp(logstd)*jax.random.normal(hk.next_rng_key(),shape=(x.shape[0], self.num_slots, self.slot_size), dtype=jnp.float32)   

        k = self.k(x)
        v = self.v(x)
        attn = None
        for i in range(T): # Iteratively applt slot attention
            slots = self.layer_norm_1(slots)

            if get_attn and i==T-1:
                slots, attn = self.attention_fn(k, self.q(slots), v, get_attn)
            else:
                slots = self.attention_fn(k, self.q(slots), v)

            slots += self.mlp(self.layer_norm_2(slots))

        return (slots, attn) if get_attn else slots

class CNNEncoder(hk.Module):
    def __init__(self, C, position_enc_fn, name=None):
        super().__init__(name=name)
        he_init = hk.initializers.VarianceScaling(scale=2.0)

        channels = C['encoder_cnn_channels']
        kernels  = C['encoder_cnn_kernels']
        strides  = C['encoder_cnn_strides']

        hidden_size = channels[-1]
        self.cnn_layers = hk.Sequential([
            hk.Conv2D(channels[0], kernels[0], stride=strides[0], padding='SAME', w_init=he_init, with_bias=True), jax.nn.relu,
            hk.Conv2D(channels[1], kernels[1], stride=strides[1], padding='SAME', w_init=he_init, with_bias=True), jax.nn.relu,
            hk.Conv2D(channels[2], kernels[2], stride=strides[2], padding='SAME', w_init=he_init, with_bias=True), jax.nn.relu,
            hk.Conv2D(hidden_size, kernels[3], stride=strides[3], padding='SAME', w_init=he_init, with_bias=True), jax.nn.relu,
        
        ])

        self.pos_embed = SoftPositionEmbed(hidden_size, C['hidden_res'], position_enc_fn)

        self.linears = hk.Sequential([ # i.e. 1x1 convolution (shared 32 neurons across all locations)
            hk.Reshape((-1, hidden_size)), # Flatten spatial dim (works with batch)
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
            hk.Linear(32, w_init=he_init), jax.nn.relu,
            hk.Linear(32, w_init=he_init),
        ])

    def __call__(self, x):
        x = self.cnn_layers(x)
        x = self.pos_embed(x)
        x = self.linears(x)
        return x

# Modified from https://github.com/google-research/google-research/tree/master/slot_attention        
class SoftPositionEmbed(hk.Module):
    """Adds soft positional embedding with learnable projection."""
    def __init__(self, hidden_size, resolution, position_enc_fn, name=None):
        """Builds the soft position embedding layer.
        args:
            hidden_size: Size of input feature dimension.
            resolution: Tuple of integers specifying width and height of grid.
        """
        super().__init__(name=name)
        self.grid = jnp.expand_dims(jnp.array(position_enc_fn(resolution)), axis=0)

        w_init = hk.initializers.VarianceScaling(scale=2.0) # He Initializing for ReLU + Linear and CNN
        self.linear = hk.Linear(hidden_size, w_init=w_init, name='soft_pos_emb_linear')

    def __call__(self, x):    
      return x + self.linear(self.grid)
