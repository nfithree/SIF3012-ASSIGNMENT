import numpy as np
import matplotlib.pyplot as plt

# Constants
h_bar = 1  # Reduced Planck constant
m = 1      # Mass of the particle
alpha = 1  # Parameter for potential well
lambda1 = 4  # Parameter for potential well

# Define the potential well function
def potential_well(x):
    return (h_bar**2 / (2 * m) * (alpha**2 * lambda1 * (lambda1 - 1) * (0.5 - 1 / np.cosh(alpha * x)**2)))

# Define the Schrödinger equation
def schrodinger(x, E, psi):
    return (-2 * m / h_bar**2 * (E - potential_well(x)) * psi)

# Define the exact eigenvalues for comparison
def eigenvalue(n):
    return (h_bar**2 / (2 * m) * (alpha**2) * (lambda1 * (lambda1 - 1) / 2 - (lambda1 - 1 - n)**2))

# Compute eigenvalues for the first three levels
energy = [eigenvalue(n) for n in range(3)]  # Use the first three eigenvalues
print("Computed Eigenvalues:", energy)

# Numerov shooting method to solve the Schrödinger equation
def numerov_method(E, x, psi_0, psi_1):
    h = x[1] - x[0]  # Grid spacing
    psi = np.zeros_like(x)
    psi[0], psi[1] = psi_0, psi_1  # Initial conditions for psi

    # Precompute q values for the entire grid
    q = 2 * m / h_bar**2 * (E - potential_well(x))
    
    # Iterate using the Numerov method
    for i in range(1, len(x) - 1):
        q_prev, q_i, q_next = q[i - 1], q[i], q[i + 1]
        # Numerov's update formula
        psi[i + 1] = (2 * (1 - 5 * h**2 * q_i / 12) * psi[i] - 
                      (1 + h**2 * q_prev / 12) * psi[i - 1]) / (1 + h**2 * q_next / 12)

    return psi

# Function to normalize the wavefunction
def normalize_wavefunction(psi):
    return psi / np.max(np.abs(psi))  # Normalize to the maximum value for visualization

# Define spatial grid
x = np.linspace(-4, 4, 1000)  # Adjust grid for the desired range

# Create figure for plotting
plt.figure(figsize=(10, 6))

# Plot the potential well
V = potential_well(x)
plt.plot(x, V, label="Potential $V(x)$", color="black", linewidth=1)

# Plot wavefunctions for each energy level
for idx, E in enumerate(energy):
    psi = numerov_method(E, x, psi_0=0, psi_1=1e-5)  # Solve the wavefunction using the Numerov method
    psi = normalize_wavefunction(psi)  # Normalize the wavefunction for better visualization
    plt.plot(x, psi + E, label=rf"$\psi(x)$ for $E = {E:.3f}$")  # Offset wavefunction by its energy

# Add labels, title, and grid to the plot
plt.title("Wavefunctions and Potential for the Schrödinger Equation")
plt.xlabel("x")
plt.ylabel(r"$\psi(x)$ \& $V(x)$")  # Use raw string for LaTeX-style labels
plt.axhline(0, color="gray", linestyle="--", linewidth=0.5)  # Horizontal axis line
plt.legend()
plt.grid(True)

# Display the plot
plt.show()
