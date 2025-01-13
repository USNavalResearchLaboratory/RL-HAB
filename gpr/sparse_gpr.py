import numpy as np
import torch
import gpytorch
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import time

# Define the true function to predict
def f(x):
    return np.sin(x) + 0.5 * np.cos(2 * x)

# Generate training data
N = 5_000  # Number of training points
X_train = np.linspace(0, 10, N).reshape(-1, 1)  # Input data
y_train = f(X_train).ravel() + 0.2 * np.random.randn(N)  # Observations with noise

# Generate test data
X_test = np.linspace(0, 10, 100).reshape(-1, 1)
y_test = f(X_test).ravel()  # Ground truth for accuracy evaluation

# Convert data to PyTorch tensors
X_train_torch = torch.tensor(X_train, dtype=torch.float32)
y_train_torch = torch.tensor(y_train, dtype=torch.float32)
X_test_torch = torch.tensor(X_test, dtype=torch.float32)

# Placeholder for results
results = {}


# Define a function to calculate mean squared error
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


### Full GPR Implementation ###
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# Train Full GPR
likelihood_gpr = gpytorch.likelihoods.GaussianLikelihood()
model_gpr = ExactGPModel(X_train_torch, y_train_torch, likelihood_gpr)

print("Training Full GPR...")
start_time = time.time()

model_gpr.train()
likelihood_gpr.train()
optimizer_gpr = torch.optim.Adam(model_gpr.parameters(), lr=0.1)
mll_gpr = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_gpr, model_gpr)

for _ in range(50):  # Training loop
    optimizer_gpr.zero_grad()
    output = model_gpr(X_train_torch)
    loss = -mll_gpr(output, y_train_torch)
    loss.backward()
    optimizer_gpr.step()

training_time_gpr = time.time() - start_time
print(f"Full GPR Training Time: {training_time_gpr:.2f} seconds")

# Predict Full GPR
model_gpr.eval()
likelihood_gpr.eval()
start_time = time.time()
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred_gpr = likelihood_gpr(model_gpr(X_test_torch))
    mean_gpr = observed_pred_gpr.mean.numpy()
    lower_gpr, upper_gpr = observed_pred_gpr.confidence_region()
prediction_time_gpr = time.time() - start_time
mse_gpr = mean_squared_error(y_test, mean_gpr)

# Store results
results['Full GPR'] = {
    'training_time': training_time_gpr,
    'prediction_time': prediction_time_gpr,
    'mse': mse_gpr,
}

# Sparse GPR Implementation with Optimized Inducing Points
class SparseGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super(SparseGPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# Train Sparse GPR
M = 100  # Number of inducing points
inducing_points = X_train_torch[::N // M]

# Perform k-means clustering to find representative inducing points
#kmeans = KMeans(n_clusters=M, random_state=42).fit(X_train)
#inducing_points = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)

likelihood_sparse = gpytorch.likelihoods.GaussianLikelihood()
model_sparse = SparseGPModel(inducing_points)

print("Training Sparse GPR...")
start_time = time.time()

model_sparse.train()
likelihood_sparse.train()
optimizer_sparse = torch.optim.Adam(model_sparse.parameters(), lr=0.1)
mll_sparse = gpytorch.mlls.VariationalELBO(likelihood_sparse, model_sparse, y_train_torch.numel())

for _ in range(50):  # Training loop
    optimizer_sparse.zero_grad()
    output = model_sparse(X_train_torch)
    loss = -mll_sparse(output, y_train_torch)
    loss.backward()
    optimizer_sparse.step()

training_time_sparse = time.time() - start_time
print(f"Sparse GPR Training Time: {training_time_sparse:.2f} seconds")

# Predict Sparse GPR
model_sparse.eval()
likelihood_sparse.eval()
start_time = time.time()
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred_sparse = likelihood_sparse(model_sparse(X_test_torch))
    mean_sparse = observed_pred_sparse.mean.numpy()
    lower_sparse, upper_sparse = observed_pred_sparse.confidence_region()
prediction_time_sparse = time.time() - start_time
mse_sparse = mean_squared_error(y_test, mean_sparse)

# Store results
results['Sparse GPR'] = {
    'training_time': training_time_sparse,
    'prediction_time': prediction_time_sparse,
    'mse': mse_sparse,
}

### Results Summary ###
print("\nComparison of Full GPR and Sparse GPR:")
for model, metrics in results.items():
    print(f"{model}:")
    print(f"  Training Time: {metrics['training_time']:.2f} seconds")
    print(f"  Prediction Time: {metrics['prediction_time']:.4f} seconds")
    print(f"  Mean Squared Error: {metrics['mse']:.6f}")

### Plot Results ###
plt.figure(figsize=(14, 6))

# Full GPR
plt.subplot(1, 2, 1)
plt.plot(X_test, f(X_test), 'r:', label='True function')
plt.scatter(X_train, y_train, s=5, color='k', label='Training data')
plt.plot(X_test, mean_gpr, 'b', label='Prediction')
plt.fill_between(X_test.ravel(), lower_gpr.numpy(), upper_gpr.numpy(), color='b', alpha=0.2, label='Confidence interval')
plt.title('Full GPR')
plt.legend()

# Sparse GPR
plt.subplot(1, 2, 2)
plt.plot(X_test, f(X_test), 'r:', label='True function')
plt.scatter(X_train, y_train, s=5, color='k', label='Training data')
plt.plot(X_test, mean_sparse, 'g', label='Prediction')
plt.fill_between(X_test.ravel(), lower_sparse.numpy(), upper_sparse.numpy(), color='g', alpha=0.2, label='Confidence interval')
plt.scatter(inducing_points.numpy(), [-2.5] * M, color='orange', s=50, label='Inducing points', marker='x')
plt.title('Sparse GPR')
plt.legend()

plt.tight_layout()
plt.show()
