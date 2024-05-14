import jax
import jax.numpy as jnp
import jax.random as jr
from jax.flatten_util import ravel_pytree


# Genetic linkage information for Drosophila taken from paper
# https://academic.oup.com/g3journal/article/2/2/287/5991485
MAFS_BY_MARKER = jnp.array([
    0.45, 0.40, 0.49, 0.35, 0.26, 0.43, 0.43, 0.37, 0.29, 0.00, 0.12, 0.29, 0.31, 
    0.31, 0.33, 0.39, 0.29, 0.46, 0.38, 0.29, 0.49, 0.31, 0.18, 0.37, 0.28, 0.29, 
    0.30, 0.43, 0.42, 0.44, 0.30, 0.30, 0.43, 0.17, 0.48, 0.00, 0.00, 0.29, 0.40, 
    0.41, 0.37, 0.22, 0.30, 0.32, 0.31, 0.29, 0.28, 0.33, 0.37, 0.46, 0.46, 0.35, 
    0.40, 0.40, 0.45, 0.43, 0.36, 0.47, 0.43, 0.32, 0.37
])
MAFS_MARKER_IDX = {
    '2': jnp.array([
        21, 12, 39, 16, 6, 53, 8, 61, 18, 58, 56, 50, 57, 41, 52, 4, 
        49, 24, 19, 54, 59, 9
    ]),
    '3': jnp.array([
        48, 43, 27, 26, 45, 14, 22, 25, 47, 38, 46, 17, 20, 13, 31, 
        60, 5, 32, 35, 3, 15, 30, 7
    ]),
    '4': jnp.array([44,]),
    'x': jnp.array([
        2, 40, 29, 28, 33, 1, 55, 51,
    ])
}
MAFS = jax.tree_util.tree_map(
    lambda x: jnp.take(MAFS_BY_MARKER, x-1), MAFS_MARKER_IDX
)
DISTANCES_2 = jnp.array([0.0, 2.8, 7.9, 8.0, 11.4, 12.6, 16.1, 17.5, 21.5,
                         24.5, 26.7, 28.0, 31.3, 32.7, 36.4, 36.8, 46.7, 48.0,
                         49.4, 51.3, 67.1, 79.0])
DISTANCES_3 = jnp.array([0.0, 3.2, 7.8, 8.0, 18.8, 20.9, 21.9, 27.8, 29.8, 
                         30.3, 32.9, 33.8, 41.8, 43.3, 45.2, 51.6, 57.2, 
                         58.6, 64.2, 74.2, 79.7, 92.1, 92.4])
DISTANCES_4 = jnp.array([0.0,])
DISTANCES_X = jnp.array([0.0, 10.1, 17.0, 20.9, 32.1, 69.7, 69.7, 73.9])
LINKAGE_MAP = {
    '2': DISTANCES_2,
    '3': DISTANCES_3,
    '4': DISTANCES_4,
    'x': DISTANCES_X,
}
# Recombination rates wrt first marker (using Haldane's mapping function)
RECOMBINATION_RATES = jax.tree_util.tree_map(
    lambda x: 0.5 * (1 - jnp.exp(-x/50.)), LINKAGE_MAP
)


def random_split_like_tree(key, target):
    if isinstance(key, int):
        key = jr.key(key)
    tree_struct = jax.tree_util.tree_structure(target)
    keys = jr.split(key, tree_struct.num_leaves)
    tree_keys = jax.tree_util.tree_unflatten(tree_struct, keys)
    return tree_keys


def generate_genotypes(key, n_population, minor_allele_frequencies=MAFS):
    if isinstance(key, int):
        key = jr.key(key)
    tree_keys = random_split_like_tree(key, minor_allele_frequencies)
    alleles = jax.tree_util.tree_map(
        lambda k, x: jr.bernoulli(
            k, 1 - x, shape=(n_population, 2, x.shape[0])
        ).astype(float), tree_keys, minor_allele_frequencies
    )
    return alleles


def generate_phenotypes(genotypes):
    genotypes = jax.tree_util.tree_map(
        lambda x: x.astype(bool), genotypes
    )
    phenotypes = jax.tree_util.tree_map(
        lambda x: (x[:,0] | x[:,1]).astype(float), genotypes
    )
    return phenotypes


def generate_frequencies(phenotypes):
    frequencies = jax.tree_util.tree_map(
        lambda x: jnp.mean(x, axis=0), phenotypes
    )
    return frequencies


def generate_mating_pairs(key, n_population):
    assert n_population > 1
    if isinstance(key, int):
        key = jr.PRNGKey(key)
    def choose_pair(key):
        return jr.choice(key, n_population, shape=(2,), replace=False)
    keys = jr.split(key, n_population)
    pairs = jax.vmap(choose_pair)(keys)
    return pairs


def generate_mating_pair_genotypes(key, genotypes):
    n_population = genotypes['2'].shape[0]
    pairs = generate_mating_pairs(key, n_population)
    mating_pair_genotypes = jax.tree_util.tree_map(
        lambda x: jnp.stack([x[pairs[:,0]], x[pairs[:,1]]]), genotypes
    )
    return mating_pair_genotypes


def generate_offsprings(key, mp_genotypes, 
                        recombination_rates=RECOMBINATION_RATES):
    if isinstance(key, int):
        key = jr.PRNGKey(key)
    key, subkey = jr.split(key)
    # Compute recombination
    tree_keys = random_split_like_tree(key, mp_genotypes)
    def _generate_recombination(key, recombination_rate, parent):
        n_parents, n_population, _, n_loci = parent.shape
        recombination = jr.bernoulli(
            key, recombination_rate, shape=(n_parents, n_population, n_loci)
        ).astype(float)
        return recombination
    recombination = jax.tree_util.tree_map(
        _generate_recombination, tree_keys, recombination_rates, mp_genotypes
    )
    # Generate offsprings
    key, subkey = jr.split(subkey)
    tree_keys = random_split_like_tree(key, mp_genotypes)
    def _generate_offsprings(key, parent, recombination):
        coin_flip = jr.bernoulli(key).astype(float)
        chromosome_select = ((recombination + coin_flip) % 2).astype(int)
        gametes = jnp.take_along_axis(
            parent, chromosome_select[:,:,None], axis=2
        )
        offsprings = jnp.transpose(gametes, (2,1,0,3)).squeeze(axis=0)
        return offsprings
    offsprings = jax.tree_util.tree_map(
        _generate_offsprings, tree_keys, mp_genotypes, recombination
    )
    return offsprings


def generate_phenotype_dataset(key, n_generations, n_population, freq=True):
    if isinstance(key, int):
        key = jr.key(key)
    if freq:
        gen_function = lambda genotypes: \
            generate_frequencies(generate_phenotypes(genotypes))
    else:
        gen_function = generate_phenotypes
    def _step(carry, key):
        key1, key2 = jr.split(key)
        curr_genotypes = carry
        phenotypes = gen_function(curr_genotypes)
        phenotypes_flattened, _ = ravel_pytree(phenotypes)
        mp_genotypes = generate_mating_pair_genotypes(key1, curr_genotypes)
        offsprings = generate_offsprings(key2, mp_genotypes)
        return offsprings, phenotypes_flattened
    key, subkey = jr.split(key)
    genotypes = generate_genotypes(key, n_population)
    keys = jr.split(subkey, n_generations)
    _, phenotypes = jax.lax.scan(_step, genotypes, keys)
    return phenotypes