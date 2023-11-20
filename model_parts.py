'''
Functions meant for use within a NumPyro model. 
All of these functions should return an Nx1 tensor / distribution
where N is the number of observations in the dataset.
'''

import numpy as np
import numpyro
import numpyro.distributions as dist

import jax.numpy as jnp

from scipy.interpolate import BSpline


def build_bym(geo_idx, geo_adj_matrix=None):
    '''
    This code builds the Besag-York-Mollie model for random effects balanced
    between spatially smoothed and independent geo unit-level effects.

    Arguments:
    ----------
    geo_idx: np.array
        Array of integers indicating which geo each observation belongs to.
    geo_adj_matrix: scipy.sparse.csr_matrix
        Adjacency matrix for the geo units. If None, then the model will
        assume that the geo units are independent.

    Returns:
    --------
    geo_effect: numpyro.distributions.Distribution
        Distribution object representing the random effect for each geo unit.
    '''        

    # Hyperparameters for the geo random effect
    sigma_geo_ind = numpyro.sample('sigma_geo_ind', dist.HalfNormal(1))
    sigma_geo_car  = numpyro.sample('sigma_geo_car', dist.HalfNormal(1))

    # Random effect for each geo
    # If we have adjacency matrix, use conditional autogression
    # Otherwise, use independent Gaussian
    n_geos = np.max(geo_idx) + 1

    with numpyro.plate('geo', n_geos):
        geo_effect_ind = numpyro.sample('geo_effect_ind', dist.Normal(0., sigma_geo_ind))

    geo_effect = geo_effect_ind

    if geo_adj_matrix is not None:

        # Check to make sure that the adjacency matrix is square, symmetric, and has no elements on the diagonal
        assert geo_adj_matrix.shape[0] == geo_adj_matrix.shape[1]
        assert np.allclose(geo_adj_matrix.diagonal(), 0)

         # Spatial autocorrelation parameter which needs to reside in (0,1)
        rho = numpyro.sample('rho', dist.Beta(2, 2))
        car_precision = sigma_geo_car**-2
        geo_effect_car = numpyro.sample('geo_effect_car', dist.CAR(0., rho, car_precision, geo_adj_matrix, is_sparse=True))

        # Combine the two effects
        geo_effect += geo_effect_car

    return geo_effect[geo_idx]

def build_factor(factor_idx, name):
    '''
    Adds a random effect for each level of a categorical variable.

    Arguments:
    ----------
    factor_idx: np.array
        Array of integers indicating which level of the factor each observation
        belongs to.
    name: str
        Label for the factor to help keep track of multiple such terms.
    
    Returns:
    --------
    factor_effect: numpyro.distributions.Distribution
        Distribution object representing the random effect for each level of
        the factor.
    '''

    # Per-category random effect
    n_levels = np.max(factor_idx) + 1
    sigma_factor = numpyro.sample(f'sigma_{name}', dist.HalfNormal(1))
    with numpyro.plate(name, n_levels):
        factor_effect = numpyro.sample(f'{name}_effect', dist.Normal(0., sigma_factor))

    return factor_effect[factor_idx]


def build_grw(times):
    '''
    Builds a Gaussian random walk for the time effect.

    Arguments:
    ----------
    times: np.array
        Array of integers indicating which time period each observation belongs
        to.

    Returns:
    --------
    time_effect: numpyro.distributions.Distribution
        Distribution object representing the random effect for each time period.
    '''

     # Gaussian random walk effect for time
    # Assumes a discrete grid of timepoints
    n_timesteps = int(np.max(times) + 1)
    sigma_grw   = numpyro.sample('sigma_grw', dist.HalfNormal(1))
    grw = numpyro.sample('grw', dist.GaussianRandomWalk(scale=sigma_grw, num_steps=n_timesteps))

    return grw[times]

def build_spline(x, n_knots, degree, n_grid_pts=100):
    '''
    Models a function of x which is a spline with a fixed number of knots.
    
    Arguments:
    ----------
    x: np.array
        Array of values for which to evaluate the spline.
    n_knots: int
        Number of knots to use in the spline.
    degree: int
        Degree of the spline.
    n_grid_pts: int
        Number of points to use when evaluating the spline on a grid for
        reference. Does not affect inference.

    Returns:
    --------
    spline_at_x: np.array
        Array of values for the spline evaluated at x.
    '''

    # We'll place the knots at equidistant percentiles of the exposure, from the min val up to the max val with the nunber
    # of knots specified by n_knots 
    x_quantiles = np.quantile(x, np.linspace(0, 1, n_knots))

    # This is to make sure all of the input domain is covered by the piecewise functions
    padded_knots = np.hstack([[x.min()] * (degree + 1), x_quantiles, [x.max()] * (degree + 1)])

    # Get the spline basis matrix
    get_basis = lambda input_var: BSpline(padded_knots, np.eye(len(padded_knots) - degree - 1), degree)(input_var)[:, 1:]

    spline_basis   = get_basis(x)
    n_spline_coefs = spline_basis.shape[1]

    sigma_spline = numpyro.sample('sigma_spline', dist.HalfNormal(1))
    spline_coefs = numpyro.sample('spline_coefs', dist.Normal(jnp.zeros(n_spline_coefs), sigma_spline*jnp.ones(n_spline_coefs) ))
    spline_term  = jnp.dot(spline_basis, spline_coefs)

    # Evaluate the spline on grid of points for reference
    # otherwise, we'd have to use the posterior samples of coefs and basis matrix to get the spline values
    # which would be relatively tedious.
    spline_grid       = np.quantile(x, np.linspace(0, 1, n_grid_pts))
    spline_grid_basis = get_basis(spline_grid)
    spline_grid_eval  = numpyro.deterministic("spline", jnp.dot(spline_grid_basis, spline_coefs))
    
    return spline_term
    
    
def build_interaction(idx1, idx2, rank):
    '''
    Builds a low-rank approximation to a full interaction term between two
    categorical variables.

    Arguments:
    ----------
    idx1: np.array
        Array of integers indicating which level of the first factor each
        observation belongs to.
    idx2: np.array
        Array of integers indicating which level of the second factor each
        observation belongs to.
    rank: int
        Rank of the approximation to use.

    Returns:
    --------
    interaction_effect: numpyro.distributions.Distribution
        Distribution object representing the random effect for each pair of levels in factor1 X factor2
    '''

    n_levels_1, n_levels_2 = np.max(idx1) + 1, np.max(idx2) + 1

    sigma_latent = numpyro.sample('sigma_latent', dist.HalfNormal(1))

    # Low-rank factorization of the interaction matrix with k factors
    latent_1 = numpyro.sample('latent_1', dist.Normal(jnp.zeros([n_levels_1, rank]), sigma_latent*jnp.ones([n_levels_1, rank])))
    latent_2 = numpyro.sample('latent_2', dist.Normal(jnp.zeros([n_levels_2, rank]), jnp.ones([n_levels_2, rank]))) # Scale is fixed so that we don'y have identifiability issues
    interaction_effect = numpyro.deterministic('interaction_effect', (latent_1 @ latent_2.T)) # Scale is fixed so that we don'y have identifiability issues
    return interaction_effect[idx1, idx2]