import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve

# Define the demand and supply functions
def demand(quantity):
    return 150 - 0.8 * quantity

def supply(quantity):
    return 20 + 0.6 * quantity

# Define the function for the difference between demand and supply
def equilibrium(quantity):
    return demand(quantity) - supply(quantity)

# Use fsolve to find the equilibrium quantity
initial_guess = 50  # Initial guess for the quantity
equilibrium_quantity = fsolve(equilibrium, initial_guess)[0]

# Calculate the equilibrium price
equilibrium_price = demand(equilibrium_quantity)

# Define the supply function with tax
tax_shift_larger = 74  # Larger shift for the supply curve due to tax
def supply_tax_adjusted_larger(quantity):
    return supply(quantity) + tax_shift_larger

# Define the function for the difference between demand and supply with tax
def equilibrium_tax(quantity):
    return demand(quantity) - supply_tax_adjusted_larger(quantity)

# Use fsolve to find the equilibrium quantity with tax
equilibrium_quantity_tax = fsolve(equilibrium_tax, initial_guess)[0]

# Calculate the equilibrium price with tax
equilibrium_price_tax = demand(equilibrium_quantity_tax)

# Set the cap level and calculate the new price under the cap
cap_quantity_adjusted = 40  # Adjust the quantity to match the new price under tax
price_cap_adjusted = np.interp(cap_quantity_adjusted, np.linspace(0, 100, 200), demand(np.linspace(0, 100, 200)))

# Set up the figure
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Emission Tax (left graph)
quantity = np.linspace(0, 100, 200)
axs[0].plot(quantity, demand(quantity), label="Demand Curve (D)", color='blue')
axs[0].plot(quantity, supply(quantity), label="Supply Curve (S)", color='red')
axs[0].plot(quantity, supply_tax_adjusted_larger(quantity), label="Supply Curve with Tax (S')", color='orange')
axs[0].axhline(y=equilibrium_price_tax, color='black', linestyle='--', label="New Price (P2)")
axs[0].axhline(y=equilibrium_price, color='grey', linestyle='--', label="Old Price (P1)")

# Labels and title for the first graph
axs[0].set_title("Emission Tax", fontsize=14)
axs[0].set_xlabel("Quantity of Product", fontsize=12)
axs[0].set_ylabel("Price/Cost", fontsize=12)
axs[0].legend(fontsize=10)
axs[0].grid(True)
axs[0].set_xticks([])  # Remove x-axis numbers
axs[0].set_yticks([])  # Remove y-axis numbers

# Emission Cap (right graph)
axs[1].plot(quantity, demand(quantity), label="Demand Curve (D)", color='blue')
axs[1].plot(quantity, supply(quantity), label="Supply Curve (S)", color='red')
axs[1].axvline(x=cap_quantity_adjusted, color='orange', linestyle='--', label="Cap Level (C)")
axs[1].axhline(y=price_cap_adjusted, color='black', linestyle='--', label="New Price (P2)")
axs[1].axhline(y=equilibrium_price, color='grey', linestyle='--', label="Old Price (P1)")

# Labels and title for the second graph
axs[1].set_title("Emission Cap", fontsize=14)
axs[1].set_xlabel("Quantity of Product", fontsize=12)
axs[1].set_ylabel("Price/Cost", fontsize=12)
axs[1].legend(fontsize=10)
axs[1].grid(True)
axs[1].set_xticks([])  # Remove x-axis numbers
axs[1].set_yticks([])  # Remove y-axis numbers

# Display the graphs
plt.tight_layout()
plt.show()