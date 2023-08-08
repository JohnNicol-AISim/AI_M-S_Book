import numpy as np
import emcee

# Define the true model
def true_model(x, theta):
    return theta * x

# Generate synthetic data with noise
np.random.seed(42)
x_observed = np.linspace(0, 10, 100)
true_theta = 2.5
y_observed = true_model(x_observed, true_theta) + np.random.normal(0, 0.5, len(x_observed))

# Define the log likelihood function for Bayesian calibration
def log_likelihood(theta, x_observed, y_observed):
    y_model = true_model(x_observed, theta)
    sigma = 0.5  # Assumed measurement error
    return -0.5 * np.sum((y_observed - y_model)**2 / sigma**2)

# Define the log prior function (uniform prior in this example)
def log_prior(theta):
    if 0.0 < theta < 5.0:
        return 0.0
    return -np.inf

# Define the log posterior function (sum of log prior and log likelihood)
def log_posterior(theta, x_observed, y_observed):
    return log_prior(theta) + log_likelihood(theta, x_observed, y_observed)

# Run MCMC sampling to find the posterior distribution of the model parameter theta
def bayesian_calibration(x_observed, y_observed, num_walkers=100, num_steps=1000):
    initial_guess = 1.0  # Initial guess for the model parameter theta
    ndim = 1  # Number of dimensions of the parameter space

    # Initialize the walkers with a small random scatter around the initial guess
    pos = initial_guess + 1e-4 * np.random.randn(num_walkers, ndim)

    # Set up the MCMC sampler
    sampler = emcee.EnsembleSampler(num_walkers, ndim, log_posterior, args=(x_observed, y_observed))

    # Run the MCMC sampling
    sampler.run_mcmc(pos, num_steps)

    # Get the samples from the posterior distribution
    samples = sampler.get_chain(discard=100, flat=True)

    # Calculate the median of the posterior distribution as the calibrated parameter
    calibrated_theta = np.median(samples)

    return calibrated_theta, samples

# Main function
def main():
    calibrated_theta, samples = bayesian_calibration(x_observed, y_observed)

    print("True Theta:", true_theta)
    print("Calibrated Theta:", calibrated_theta)

if __name__ == "__main__":
    main()
