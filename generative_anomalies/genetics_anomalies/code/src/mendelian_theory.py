import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln


def log_binom(x, y):
  return gammaln(x + 1) - gammaln(y + 1) - gammaln(x - y + 1)


def log_transition(prev_phen, next_phen, n_population):
    log_p = log_binom(n_population, next_phen * n_population)
    log_p += next_phen * n_population * jnp.log(
        4/3 * prev_phen * (1 - 1/3 * prev_phen)
    )
    log_p += (1 - next_phen) * n_population * jnp.log(
        (2/3 * prev_phen - 1) ** 2
    )
    return log_p


def compute_mendelian_log_likelihood(phenotypes, n_population):
    def _step(carry, t):
        prev_phens, next_phens = phenotypes[t], phenotypes[t+1]
        log_ps = jax.vmap(
            log_transition, (0, 0, None)
        )(prev_phens, next_phens, n_population)
        log_p = jnp.sum(log_ps)
        return None, log_p
    _, log_ps = jax.lax.scan(_step, None, jnp.arange(phenotypes.shape[0] - 1))
    log_p = jnp.sum(log_ps)
    return log_p