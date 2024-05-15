import argparse
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt

from generative_anomalies.genetics_anomalies import settings
from generative_anomalies.models import vae


def save_model_and_plot_losses(ckpt_dir, vae_params):
    # Save model
    with open(Path(ckpt_dir, "params_enc.npy"), "wb") as f:
        jnp.save(f, vae_params["params_enc"])
    with open(Path(ckpt_dir, "params_dec.npy"), "wb") as f:
        jnp.save(f, vae_params["params_dec"])
    with open(Path(ckpt_dir, "mu_mean.npy"), "wb") as f:
        jnp.save(f, vae_params["mu_mean"])
    with open(Path(ckpt_dir, "mu_std.npy"), "wb") as f:
        jnp.save(f, vae_params["mu_std"])
    with open(Path(ckpt_dir, "sigmasq_mean.npy"), "wb") as f:
        jnp.save(f, vae_params["sigmasq_mean"])
    with open(Path(ckpt_dir, "sigmasq_std.npy"), "wb") as f:
        jnp.save(f, vae_params["sigmasq_std"])
    with open(Path(ckpt_dir, "losses.npy"), "wb") as f:
        jnp.save(f, vae_params["losses"])
    with open(Path(ckpt_dir, "losses_rec.npy"), "wb") as f:
        jnp.save(f, vae_params["losses_rec"])
    with open(Path(ckpt_dir, "losses_kl.npy"), "wb") as f:
        jnp.save(f, vae_params["losses_kl"])
        
    # Plot losses
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(vae_params["losses"], label="Total loss")
    ax.legend()
    fig.savefig(Path(ckpt_dir, "losses.png"))
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(vae_params["losses_rec"], label="Reconstruction loss")
    ax.legend()
    fig.savefig(Path(ckpt_dir, "losses_rec.png"))
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(vae_params["losses_kl"], label="KL loss")
    ax.legend()
    fig.savefig(Path(ckpt_dir, "losses_kl.png"))


def main(args):
    # Load dataset
    X_tr = jnp.load(Path(settings.data_path, "X_tr.npy"))
    
    # Train model
    vae_params = vae.train_vae(
        X_tr, args.beta, args.z_dim, args.seed, args.n_epochs, args.batch_size,
        args.hidden_width, args.hidden_depth, args.lr_init, args.lr_peak,
        args.lr_end, beta_scheduler_type=args.beta_scheduler
    )
    
    # Save model
    gen_ckpt_dir = Path(settings.result_path, "checkpoints", "vae")
    gen_ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_model_and_plot_losses(gen_ckpt_dir, vae_params)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Parameters for VAE
    parser.add_argument("--z_dim", type=int, default=512) # latent dim
    parser.add_argument("--beta", type=float, default=0.01) # KL-div reg. weight
    parser.add_argument("--hidden_width", type=int, default=100) # hidden layer width
    parser.add_argument("--hidden_depth", type=int, default=4) # hidden layer depth
    parser.add_argument("--lr_init", type=float, default=1e-7) # initial learning rate
    parser.add_argument("--lr_peak", type=float, default=1e-4) # peak learning rate
    parser.add_argument("--lr_end", type=float, default=1e-7) # end learning rate
    parser.add_argument("--beta_scheduler", type=str, default="constant",
                        choices=["constant", "linear", "cosine",
                                 "warmup_cosine", "cyclical"],)
    
    # Specify training parameters
    parser.add_argument("--seed", type=int, default=0) # random seed
    parser.add_argument("--n_epochs", type=int, default=100) # number of epochs to train
    parser.add_argument("--batch_size", type=int, default=512) # batch size
    
    args = parser.parse_args()
    main(args)