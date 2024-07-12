
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import numpy as np

import matplotlib.pyplot as plt

import time as T

from copy import deepcopy


def train_mlp_network(X, 
                      y, 
                      n_blocks, 
                      n_units_per_layer, 
                      learning_rate, 
                      batch_size, 
                      n_epochs, 
                      random_seed, 
                      plot_loss=True, 
                      print_loss=True
):

    if random_seed is not None:
        torch.manual_seed(random_seed)
    
    latent_dim = 0
    n_blocks = n_blocks
    n_units_per_layer = n_units_per_layer
    mlp = MLP(latent_dim=latent_dim, n_blocks=n_blocks, n_units_per_layer=n_units_per_layer)

    # Define loss function and optimizer
    criterion = nn.MSELoss()  # Mean Squared Error loss for regression
    optimizer = optim.Adam(mlp.parameters(), lr=learning_rate)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize inputs
    input_scaler = StandardScaler()
    X_train = input_scaler.fit_transform(X_train)
    X_test = input_scaler.transform(X_test)

    # Normalize outputs
    output_scaler = StandardScaler()
    y_train = output_scaler.fit_transform(y_train)
    y_test = output_scaler.transform(y_test)

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)  # Reshape y to be a column vector
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)  # Reshape y to be a column vector

    # Define batch size
    batch_size = batch_size

    # Training loop with batch training
    train_losses, test_losses = [], []
    best_test_loss = np.inf
    for epoch in range(n_epochs):
        t = T.time()
        # Shuffle the training data and split it into mini-batches
        indices = torch.randperm(X_train_tensor.shape[0])
        for i in range(0, X_train_tensor.shape[0], batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_inputs = X_train_tensor[batch_indices]
            batch_targets = y_train_tensor[batch_indices]

            optimizer.zero_grad()
            batch_outputs = mlp(batch_inputs)
            batch_loss = criterion(batch_outputs, batch_targets)
            batch_loss.backward()
            optimizer.step()

        with torch.no_grad():
            test_outputs = mlp(X_test_tensor)
            test_loss = criterion(test_outputs, y_test_tensor)
            train_outputs = mlp(X_train_tensor)
            train_loss = criterion(train_outputs, y_train_tensor)

        # train_losses.append(train_loss.item())
        test_losses.append(test_loss.item())

        if test_loss < best_test_loss:
            best_state_dict = deepcopy(mlp.state_dict())
            best_test_loss = test_loss

        if print_loss and (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{n_epochs}], Train Loss: {train_loss.item():.8f}, Test Loss: {test_loss.item():.8f}, Epoch runtime: {np.round(T.time()-t, 5)} s.')
        


    mlp.load_state_dict(best_state_dict)
    
    if plot_loss:
        plt.figure(figsize=(5,5))
        plt.plot(train_losses, label='train')
        plt.plot(test_losses, label='test')
        plt.yscale('log')
        plt.legend()
        plt.show()
    
    return mlp, input_scaler, output_scaler


class MLP(nn.Module):
    
    def __init__(self, latent_dim, n_blocks, n_units_per_layer):

        assert n_blocks >= 3
        
        super(MLP, self).__init__()
        self.latent_dim = latent_dim
        self.n_units_per_layer = n_units_per_layer
        self.n_blocks = n_blocks

        blocks = [
            nn.Sequential(
            nn.Linear(3+self.latent_dim, self.n_units_per_layer),
            nn.LeakyReLU(),
            nn.Linear(self.n_units_per_layer, self.n_units_per_layer),
            nn.LeakyReLU(),
        )
        ]
        blocks += [
            nn.Sequential(
                nn.Linear(self.n_units_per_layer+3+self.latent_dim, self.n_units_per_layer),
                nn.LeakyReLU(),
                nn.Linear(self.n_units_per_layer, self.n_units_per_layer),
                nn.LeakyReLU(),
            ) for i in range(self.n_blocks-2)
        ]
        blocks.append(
            nn.Sequential(
                nn.Linear(self.n_units_per_layer+3+self.latent_dim, self.n_units_per_layer),
                nn.LeakyReLU(),
                nn.Linear(self.n_units_per_layer, self.n_units_per_layer//2),
                nn.LeakyReLU(),
                nn.Linear(self.n_units_per_layer//2, 1),
            )
        )

        self.blocks = nn.ModuleList(blocks)
        
        
    def forward(self, x):
        x_ = self.blocks[0](x)
        for i in range(1, self.n_blocks):
            x_ = self.blocks[i](torch.cat((x_, x), dim=1))
        return x_
    
class NeuralUdf():
    def __init__(self, model, input_scaler=None, output_scaler=None):
        self.model=model
        self.input_scaler=input_scaler
        self.output_scaler=output_scaler
    def __call__(self, X, with_grad=False):
        if isinstance(X, np.ndarray):
            X = torch.tensor(X)
        if self.input_scaler is not None:
            torch_scale = torch.tensor(self.input_scaler.scale_).reshape(1,3)
            torch_mean = torch.tensor(self.input_scaler.mean_).reshape(1,3)
            X = (X - torch_mean) / torch_scale
        y = self.model(X.float())
        if self.output_scaler is not None:
            torch_scale = torch.tensor(self.output_scaler.scale_)
            torch_mean = torch.tensor(self.output_scaler.mean_)
            y = torch_scale*y + torch_mean
        if with_grad == False:
            y = y.detach().numpy()
        return y
    
class NeuralSquaredError():
    def __init__(self, udf, neural_udf):
        self.udf=udf
        self.neural_udf=neural_udf
        
    def __call__(self, X):
        y_true = self.udf(X).flatten()
        y_pred = self.neural_udf(X).flatten()
        return (y_true-y_pred)**2