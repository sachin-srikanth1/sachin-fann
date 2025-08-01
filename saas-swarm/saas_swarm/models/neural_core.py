"""
TinyNeuralNetwork module for SaaS-Swarm platform.

Implements a lightweight neural network using NumPy for agent inference
and online learning capabilities.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
import json
import os


class TinyNeuralNetwork:
    """
    Lightweight neural network implementation using NumPy.
    
    Features:
    - Single or dual-layer architecture
    - Forward pass for inference
    - Online training with backpropagation
    - Model saving/loading
    - Efficient memory usage
    """
    
    def __init__(self, input_size: int, hidden_size: int = 64, 
                 output_size: int = 10, learning_rate: float = 0.01):
        """
        Initialize the neural network.
        
        Args:
            input_size: Size of input layer
            hidden_size: Size of hidden layer
            output_size: Size of output layer
            learning_rate: Learning rate for training
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Initialize weights and biases
        self.weights1 = np.random.randn(input_size, hidden_size) * 0.01
        self.biases1 = np.zeros((1, hidden_size))
        self.weights2 = np.random.randn(hidden_size, output_size) * 0.01
        self.biases2 = np.zeros((1, output_size))
        
        # Training history
        self.training_history = []
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the network.
        
        Args:
            x: Input data (batch_size, input_size)
            
        Returns:
            Output predictions (batch_size, output_size)
        """
        # Convert list to numpy array if needed
        if isinstance(x, list):
            x = np.array(x, dtype=np.float32)
        
        # Ensure input is 2D
        if x.ndim == 1:
            x = x.reshape(1, -1)
            
        # First layer
        self.z1 = np.dot(x, self.weights1) + self.biases1
        self.a1 = self._relu(self.z1)
        
        # Second layer
        self.z2 = np.dot(self.a1, self.weights2) + self.biases2
        self.output = self._softmax(self.z2)
        
        return self.output
        
    def train_step(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Perform one training step with backpropagation.
        
        Args:
            x: Input data (batch_size, input_size)
            y: Target data (batch_size, output_size)
            
        Returns:
            Loss value
        """
        # Convert inputs to numpy arrays if needed
        if isinstance(x, list):
            x = np.array(x, dtype=np.float32)
        if isinstance(y, list):
            y = np.array(y, dtype=np.float32)
        
        # Forward pass
        self.forward(x)
        
        # Calculate loss
        loss = self._cross_entropy_loss(self.output, y)
        
        # Backward pass
        self._backward(x, y)
        
        # Record training step
        self.training_history.append({
            'loss': loss,
            'learning_rate': self.learning_rate
        })
        
        return loss
        
    def _backward(self, x: np.ndarray, y: np.ndarray):
        """Backward pass to compute gradients and update weights."""
        # Ensure x is 2D
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if y.ndim == 1:
            y = y.reshape(1, -1)
            
        batch_size = x.shape[0]
        
        # Gradient of loss with respect to output
        d_output = self.output - y
        
        # Gradient of loss with respect to weights2
        d_weights2 = np.dot(self.a1.T, d_output)
        d_biases2 = np.sum(d_output, axis=0, keepdims=True)
        
        # Gradient of loss with respect to hidden layer
        d_hidden = np.dot(d_output, self.weights2.T)
        d_hidden_relu = d_hidden * self._relu_derivative(self.z1)
        
        # Gradient of loss with respect to weights1
        d_weights1 = np.dot(x.T, d_hidden_relu)
        d_biases1 = np.sum(d_hidden_relu, axis=0, keepdims=True)
        
        # Update weights and biases
        self.weights2 -= self.learning_rate * d_weights2
        self.biases2 -= self.learning_rate * d_biases2
        self.weights1 -= self.learning_rate * d_weights1
        self.biases1 -= self.learning_rate * d_biases1
        
    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation function."""
        return np.maximum(0, x)
        
    def _relu_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of ReLU activation function."""
        return np.where(x > 0, 1, 0)
        
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax activation function."""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
        
    def _cross_entropy_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Cross-entropy loss function."""
        epsilon = 1e-15  # Small value to prevent log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Make predictions without training."""
        return self.forward(x)
        
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the neural network."""
        return {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'learning_rate': self.learning_rate,
            'training_steps': len(self.training_history),
            'average_loss': np.mean([step['loss'] for step in self.training_history[-10:]]) if self.training_history else 0.0
        }
        
    def save(self, filepath: str):
        """Save the neural network to a file."""
        model_data = {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'learning_rate': self.learning_rate,
            'weights1': self.weights1.tolist(),
            'biases1': self.biases1.tolist(),
            'weights2': self.weights2.tolist(),
            'biases2': self.biases2.tolist(),
            'training_history': self.training_history
        }
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f)
            
    def load(self, filepath: str):
        """Load the neural network from a file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
            
        with open(filepath, 'r') as f:
            model_data = json.load(f)
            
        self.input_size = model_data['input_size']
        self.hidden_size = model_data['hidden_size']
        self.output_size = model_data['output_size']
        self.learning_rate = model_data['learning_rate']
        
        self.weights1 = np.array(model_data['weights1'])
        self.biases1 = np.array(model_data['biases1'])
        self.weights2 = np.array(model_data['weights2'])
        self.biases2 = np.array(model_data['biases2'])
        
        self.training_history = model_data.get('training_history', [])
        
    def reset(self):
        """Reset the neural network to initial state."""
        self.weights1 = np.random.randn(self.input_size, self.hidden_size) * 0.01
        self.biases1 = np.zeros((1, self.hidden_size))
        self.weights2 = np.random.randn(self.hidden_size, self.output_size) * 0.01
        self.biases2 = np.zeros((1, self.output_size))
        self.training_history = []
        
    def get_parameter_count(self) -> int:
        """Get the total number of parameters in the network."""
        return (self.input_size * self.hidden_size + self.hidden_size + 
                self.hidden_size * self.output_size + self.output_size)
        
    def get_memory_usage(self) -> int:
        """Get approximate memory usage in bytes."""
        param_count = self.get_parameter_count()
        return param_count * 4  # Assuming float32 (4 bytes per parameter)


class DualLayerNeuralNetwork(TinyNeuralNetwork):
    """
    Dual-layer neural network with an additional hidden layer.
    """
    
    def __init__(self, input_size: int, hidden_size1: int = 64, 
                 hidden_size2: int = 32, output_size: int = 10, 
                 learning_rate: float = 0.01):
        """
        Initialize the dual-layer neural network.
        
        Args:
            input_size: Size of input layer
            hidden_size1: Size of first hidden layer
            hidden_size2: Size of second hidden layer
            output_size: Size of output layer
            learning_rate: Learning rate for training
        """
        super().__init__(input_size, hidden_size1, output_size, learning_rate)
        self.hidden_size2 = hidden_size2
        
        # Additional layer weights and biases
        self.weights2 = np.random.randn(hidden_size1, hidden_size2) * 0.01
        self.biases2 = np.zeros((1, hidden_size2))
        self.weights3 = np.random.randn(hidden_size2, output_size) * 0.01
        self.biases3 = np.zeros((1, output_size))
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through the dual-layer network."""
        if x.ndim == 1:
            x = x.reshape(1, -1)
            
        # First layer
        self.z1 = np.dot(x, self.weights1) + self.biases1
        self.a1 = self._relu(self.z1)
        
        # Second layer
        self.z2 = np.dot(self.a1, self.weights2) + self.biases2
        self.a2 = self._relu(self.z2)
        
        # Third layer (output)
        self.z3 = np.dot(self.a2, self.weights3) + self.biases3
        self.output = self._softmax(self.z3)
        
        return self.output
        
    def _backward(self, x: np.ndarray, y: np.ndarray):
        """Backward pass for dual-layer network."""
        batch_size = x.shape[0]
        
        # Gradient of loss with respect to output
        d_output = self.output - y
        
        # Gradient of loss with respect to weights3
        d_weights3 = np.dot(self.a2.T, d_output)
        d_biases3 = np.sum(d_output, axis=0, keepdims=True)
        
        # Gradient of loss with respect to second hidden layer
        d_hidden2 = np.dot(d_output, self.weights3.T)
        d_hidden2_relu = d_hidden2 * self._relu_derivative(self.z2)
        
        # Gradient of loss with respect to weights2
        d_weights2 = np.dot(self.a1.T, d_hidden2_relu)
        d_biases2 = np.sum(d_hidden2_relu, axis=0, keepdims=True)
        
        # Gradient of loss with respect to first hidden layer
        d_hidden1 = np.dot(d_hidden2_relu, self.weights2.T)
        d_hidden1_relu = d_hidden1 * self._relu_derivative(self.z1)
        
        # Gradient of loss with respect to weights1
        d_weights1 = np.dot(x.T, d_hidden1_relu)
        d_biases1 = np.sum(d_hidden1_relu, axis=0, keepdims=True)
        
        # Update weights and biases
        self.weights3 -= self.learning_rate * d_weights3
        self.biases3 -= self.learning_rate * d_biases3
        self.weights2 -= self.learning_rate * d_weights2
        self.biases2 -= self.learning_rate * d_biases2
        self.weights1 -= self.learning_rate * d_weights1
        self.biases1 -= self.learning_rate * d_biases1 