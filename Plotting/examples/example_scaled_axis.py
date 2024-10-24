import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.constants import g, R

def pressure_to_altitude(pressure, sea_level_pressure=101325, temperature_lapse_rate=-0.0065, sea_level_temperature=288.15):
    """
    Convert pressure to altitude using the barometric formula.

    Parameters:
    - pressure: The pressure in Pascals.
    - sea_level_pressure: The pressure at sea level in Pascals (default is 101325 Pa).
    - temperature_lapse_rate: The temperature lapse rate in K/m (default is -0.0065 K/m).
    - sea_level_temperature: The temperature at sea level in Kelvin (default is 288.15 K).

    Returns:
    - The altitude in meters.
    """
    altitude = (sea_level_temperature / temperature_lapse_rate) * ((pressure / sea_level_pressure) ** (-temperature_lapse_rate * R / g) - 1)
    return altitude

# Sample data
x = np.linspace(0, 10, 100)
y = np.sin(x)
pressure = 101325 * np.exp(-x / 8)  # Simulated pressure values in Pa

# Convert pressure to altitude for tick labels
z_ticks_pressure = np.array([80000, 90000, 101325])
z_ticks_altitude = pressure_to_altitude(z_ticks_pressure)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plotting the data using the original pressure values
ax.plot(x, y, pressure, label='Trajectory')

# Setting the Z ticks to pressure and labels to corresponding altitude
ax.set_zticks(z_ticks_pressure)
ax.set_zticklabels([f'{alt:.2f} m' for alt in z_ticks_altitude])

# Optional: set labels for better understanding
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Pressure (Pa) / Altitude (m)')
ax.legend()

plt.show()
