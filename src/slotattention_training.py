import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
tf.config.experimental.set_visible_devices([], "GPU")
import haiku as hk
import optax
from functools import partial
import tensorflow_datasets as tfds


from .tetrominoes_loader import dataset as tetrominoes_dataloader
from .slotattention_model import SlotAttentionAE

cfg = {
    "mlp_hidden_size": 128,
    "spatial_broadcast_dims": [35,35],
    "hidden_res": [35,35],
    "slots": 4,
    "slot_size": 32,
    "attention_eps": 1E-8,
    "encoder_cnn_channels": [32,32,32,32],
    "encoder_cnn_kernels": [5,5,5,5],
    "encoder_cnn_strides": [1,1,1,1],
    "decoder_cnn_channels": [32,32,32],
    "decoder_cnn_kernels": [5,5,5,5] ,
    "decoder_cnn_strides": [1,1,1,1] ,
    "extra_deconv_layers": 0,
    "debug": 0 ,
    "adam_lr": 4E-4 ,
    "adam_beta_1": 0.9 ,
    "adam_beta_2": 0.999 ,
    "adam_eps": 1E-8 ,
    "lr_decay_rate": 0.5 ,
    "warmup_iter": 1E+4 ,
    "decay_steps": 1E+5 ,
    "batch_size": 64 ,
    "train_steps": 5E+5 ,
    "rng_seed": 17 
}


def train_model(ds, attention_fn, position_enc_fn):
    logdir = "./logs/"

    global net, opt
    np.random.seed(cfg['rng_seed'])
    tf.random.set_seed(cfg['rng_seed']) # For loading / shuffling of dset
    rng_seq = hk.PRNGSequence(cfg['rng_seed'])

    test_image = jnp.asarray(next(ds)[-1], dtype=jnp.float32)[None,:,:,:]/255.0
    reco_key = jax.random.PRNGKey(cfg['rng_seed']+1) # Naughty things will happen if we try to adjust the 

    # Initialize network and optimizer
    net = hk.transform(partial(forward_fn, attention_fn=attention_fn, position_enc_fn=position_enc_fn, cfg=cfg))
    params = net.init(next(rng_seq), test_image)
    print("Network Initialized")
    print("Model has " +str(hk.data_structures.tree_size(params))+" parameters")

    opt =  get_optimizer(cfg)
    opt_state = opt.init(params)

    # Train
    file_writer = tf.summary.create_file_writer(logdir)
    with file_writer.as_default():
        tf.summary.image("Training Source", test_image, step=0)
    test_image = (test_image-0.5)*2

    step = 0
    print("Training Starting")
    while step < 5E+5:
        step += 1
        batch = next(ds)
        batch = ((jnp.asarray(batch, dtype=jnp.float32)/255.)-0.5)*2.
        # Do SGD on a batch of training examples.
        loss, params, opt_state = update(params, next(rng_seq), opt_state, batch)

        # Apply model on test sequence for tensorboard
        if step % 500 == 0:         
            # Log a reconstruction and accompanying attention masks 
            reco, attn = net.apply(params, reco_key, (test_image, True))
            reco = (reco/2.)+0.5
            # Horitontally stack masks
            attn = np.expand_dims(np.hstack(list(attn[0].T.reshape(4,35,35))), axis=(0,-1)) 

            with file_writer.as_default():
                tf.summary.image("Training Reco", reco, step=step)
                tf.summary.image("Attention Masks", attn, step=step)

        if step % 100 == 0:
            with file_writer.as_default():
                tf.summary.scalar('loss', loss, step=step)

@jax.jit
def update(params, rng_key, opt_state, batch):
    """Learning rule (stochastic gradient descent)."""
    loss_val, grads = jax.value_and_grad(loss)(params, rng_key, batch)
    updates, opt_state = opt.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return loss_val, new_params, opt_state

def loss(params, rng_key, batch):
    """cfgompute the loss of the network, including L2."""
    reco = net.apply(params, rng_key, batch)
    return jnp.mean((batch-reco)**2)

def forward_fn(x, attention_fn=None, position_enc_fn=None, cfg=None):
    get_attn=False
    if isinstance(x, tuple):
        x, get_attn  = x
    model = SlotAttentionAE(cfg, attention_fn, position_enc_fn)
    return model(x, get_attn=get_attn)

def get_optimizer(config):
    warm_up_poly = optax.polynomial_schedule(init_value=1/config['warmup_iter'], end_value=1,
                                             power=1, transition_steps=config['warmup_iter'])
    exp_decay = optax.exponential_decay(init_value=config['adam_lr'], transition_steps=config['decay_steps'],
                                        decay_rate=config['lr_decay_rate'], transition_begin=0)#config['warmup_iter']) 
    opt = optax.chain(
        # clip_by_global_norm(max_norm),
        optax.scale_by_adam(b1=config['adam_beta_1'],b2=config['adam_beta_2'],eps=config['adam_eps']),
        optax.scale_by_schedule(warm_up_poly),
        optax.scale_by_schedule(exp_decay), 
        optax.scale(-1)
    )
    return opt

def load_data(data_path, batch_size=20, training=True):
    parallel_map_calls = tf.data.experimental.AUTOTUNE
    dataset = tetrominoes_dataloader(data_path, get_masks=False, map_parallel_calls=parallel_map_calls)
  
    if training:
        dataset = dataset.shuffle(10*batch_size, seed=cfg['rng_seed'],reshuffle_each_iteration=True).repeat()

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return iter(tfds.as_numpy(dataset)) 

