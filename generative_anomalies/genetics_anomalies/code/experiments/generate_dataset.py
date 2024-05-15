import argparse
from pathlib import Path

import jax.numpy as jnp
import jax.random as jr
import tqdm

from generative_anomalies.genetics_anomalies.code.src import data_generator
from generative_anomalies.genetics_anomalies import settings


def generate_dataset(key, n_samples, n_generations, n_population):
    X = []
    for _ in tqdm.trange(n_samples):
        key, subkey = jr.split(key)
        phenotypes = data_generator.generate_phenotype_dataset(
            subkey, n_generations, n_population
        )
        X.append(phenotypes)
    X = jnp.stack(X)
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
    
    print("Generating test data...")
    X_te = generate_dataset(
        key_te, args.n_test, args.n_generations, args.n_population
    )
    jnp.save(Path(settings.data_path, "X_te.npy"), X_te)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_train", type=int, default=100_000)
    parser.add_argument("--n_test", type=int, default=100_000)
    parser.add_argument("--n_generations", type=int, default=50)
    parser.add_argument("--n_population", type=int, default=1_000)
    
    args = parser.parse_args()
    main(args)