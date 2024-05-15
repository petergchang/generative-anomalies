import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
from jax_tqdm import scan_tqdm

from generative_anomalies.genetics_anomalies.code.src import data_generator
from generative_anomalies.genetics_anomalies import settings


def generate_dataset(key, n_samples, n_generations, n_population):
    @scan_tqdm(n_samples)
    def _step(carry, i):
        key_curr = jr.fold_in(key, i)
        phenotypes = data_generator.generate_phenotype_dataset(
            key_curr, n_generations, n_population
        )
        return None, phenotypes
    _, X = jax.lax.scan(_step, None, jnp.arange(n_samples))
    return X


def main(args):
    settings.data_path.mkdir(parents=True, exist_ok=True)
    key = jr.key(args.seed)
    key_tr, key_te = jr.split(key)
    print("Generating training data...")
    X_tr = generate_dataset(
        key_tr, args.n_train, args.n_generations, args.n_population
    )
    jnp.save(Path(settings.data_path, "X_tr.npy"), X_tr)
    
    keys_te = jr.split(key_te, args.n_test)
    print("Generating test data...")
    X_te = generate_dataset(
        keys_te, args.n_test, args.n_generations, args.n_population
    )
    jnp.save(Path(settings.data_path, "X_te.npy"), X_te)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_train", type=int, default=10_000)
    parser.add_argument("--n_test", type=int, default=10_000)
    parser.add_argument("--n_generations", type=int, default=10)
    parser.add_argument("--n_population", type=int, default=10_000)
    
    args = parser.parse_args()
    main(args)