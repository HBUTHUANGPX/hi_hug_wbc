import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import Normal

# Define the function C(phi_t_i)
def C(phi_t_i, sigma):
    # Create normal distributions for the CDF calculations
    normal_dist = Normal(loc=0, scale=1)

    # Calculate the components of the function
    term1 = normal_dist.cdf(phi_t_i / sigma) * (1 - normal_dist.cdf((phi_t_i - 0.5) / sigma))
    term2 = normal_dist.cdf((phi_t_i - 1) / sigma) * (1 - normal_dist.cdf((phi_t_i - 1.5) / sigma))
    
    return term1 + term2

# Generate values for phi_t_i and choose sigma
phi_t_i_values = torch.linspace(0, 1, 1000)  # range of phi_t_i
sigma = .05  # Example standard deviation value

# Compute the values of C(phi_t_i)
C_values = C(phi_t_i_values, sigma)

# Plot the function
plt.figure(figsize=(8, 6))
plt.plot(phi_t_i_values.numpy(), C_values.numpy(), label=r'$C(\phi_{t,i})$', color='b')
plt.xlabel(r'$\phi_{t,i}$', fontsize=14)
plt.ylabel(r'$C(\phi_{t,i})$', fontsize=14)
plt.title(r'Plot of $C(\phi_{t,i})$', fontsize=16)
plt.grid(True)
plt.legend()
plt.show()
