import numpy as np
import matplotlib.pyplot as plt
from my_python_package.linalg_interp import spline_function

# Load data
water_data = np.loadtxt('data/water_density_vs_temp_usgs.txt')
air_data = np.loadtxt('data/air_density_vs_temp_eng_toolbox.txt')

# Format data
water_temp, water_density = water_data[:, 0], water_data[:, 1]
air_temp, air_density = air_data[:, 0], air_data[:, 1]


# Generate spline functions for orders 1,2,3
orders = [1, 2, 3]

water_splines = {k: spline_function(water_temp, water_density, order=k) for k in orders}
air_splines   = {k: spline_function(air_temp, air_density, order=k) for k in orders}

# Generate 100 equally spaced temperature values for interpolation
water_temp_interp = np.linspace(water_temp.min(), water_temp.max(), 100)
air_temp_interp   = np.linspace(air_temp.min(), air_temp.max(), 100)

# Plot data points and spline fits
fig, axes = plt.subplots(3, 2, figsize=(12, 10))
axes = axes.flatten()  # easier indexing

for i, order in enumerate(orders):
    # Water
    axes[i*2].scatter(water_temp, water_density, color='blue', label='Data')
    axes[i*2].plot(water_temp_interp, water_splines[order](water_temp_interp), color='red', label=f'Order {order} spline')
    axes[i*2].set_title(f'Water Density - Order {order}')
    axes[i*2].set_xlabel('Temperature')
    axes[i*2].set_ylabel('Density')
    axes[i*2].legend()
    axes[i*2].grid(True)
    
    # Air
    axes[i*2+1].scatter(air_temp, air_density, color='green', label='Data')
    axes[i*2+1].plot(air_temp_interp, air_splines[order](air_temp_interp), color='orange', label=f'Order {order} spline')
    axes[i*2+1].set_title(f'Air Density - Order {order}')
    axes[i*2+1].set_xlabel('Temperature')
    axes[i*2+1].set_ylabel('Density')
    axes[i*2+1].legend()
    axes[i*2+1].grid(True)

plt.tight_layout()
# Save figure for report
plt.savefig('Figures/density_spline_plots.png')
plt.show()
