import numpy as np

# Create a fixed generator for ordered (reproducible) results
ordered_rng = np.random.default_rng(42)  # Seed with a fixed value

# Generate ordered random numbers
ordered_random = ordered_rng.uniform(0, 1, size=1)
print("Ordered random numbers:", ordered_random)
ordered_random = ordered_rng.uniform(0, 1, size=1)
print("Ordered random numbers:", ordered_random)
ordered_random = ordered_rng.uniform(0, 1, size=1)
print("Ordered random numbers:", ordered_random)

# Create a new generator for truly random numbers (using default state)
truly_random_rng = np.random.default_rng()  # No seed, truly random

# Generate truly random numbers
truly_random = truly_random_rng.uniform(0, 1, size=1)
print("Truly random numbers:", truly_random)
truly_random = truly_random_rng.uniform(0, 1, size=1)
print("Truly random numbers:", truly_random)
truly_random = truly_random_rng.uniform(0, 1, size=1)
print("Truly random numbers:", truly_random)