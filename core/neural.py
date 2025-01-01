import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import json

@dataclass
class NetworkConfig:
    """Neural network configuration"""
    input_size: int
    hidden_sizes: List[int]
    output_size: int
    activation: str = 'relu'
    dropout_rate: float = 0.1
    use_batch_norm: bool = True

class NeuralProcessor(nn.Module):
    """Neural network for processing agent inputs and generating actions"""
    
    def __init__(self, config: NetworkConfig):
        super().__init__()
        self.config = config
        
        # Build network layers
        layers = []
        prev_size = config.input_size
        
        for hidden_size in config.hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                self._get_activation(config.activation),
                nn.Dropout(config.dropout_rate)
            ])
            if config.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            prev_size = hidden_size
            
        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_size, config.output_size)
        
        # Initialize attention mechanism
        self.attention = SelfAttention(config.hidden_sizes[-1])
        
        # Memory buffer for experience replay
        self.memory_buffer = []
        self.max_memory_size = 10000
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network"""
        # Process through hidden layers
        hidden = self.hidden_layers(x)
        
        # Apply attention
        attended, attention_weights = self.attention(hidden)
        
        # Generate output
        output = self.output_layer(attended)
        
        return output, attention_weights
        
    def _get_activation(self, activation_name: str) -> nn.Module:
        """Get activation function by name"""
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU()
        }
        return activations.get(activation_name, nn.ReLU())
        
    def store_experience(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        next_state: torch.Tensor
    ) -> None:
        """Store experience in memory buffer"""
        experience = (state, action, reward, next_state)
        self.memory_buffer.append(experience)
        
        if len(self.memory_buffer) > self.max_memory_size:
            self.memory_buffer.pop(0)
            
    def sample_experience(self, batch_size: int) -> List[Tuple]:
        """Sample batch of experiences for training"""
        return random.sample(self.memory_buffer, min(batch_size, len(self.memory_buffer)))
        
    def update(self, experiences: List[Tuple], learning_rate: float = 0.001) -> float:
        """Update network based on experiences"""
        if not experiences:
            return 0.0
            
        # Unpack experiences
        states, actions, rewards, next_states = zip(*experiences)
        
        # Convert to tensors
        states = torch.stack(states)
        actions = torch.stack(actions)
        rewards = torch.tensor(rewards)
        next_states = torch.stack(next_states)
        
        # Compute target Q-values
        with torch.no_grad():
            next_q_values, _ = self(next_states)
            targets = rewards + 0.99 * next_q_values.max(1)[0]
            
        # Compute current Q-values
        q_values, attention_weights = self(states)
        current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze()
        
        # Compute loss
        loss = F.smooth_l1_loss(current_q, targets)
        
        # Update network
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return float(loss)
        
class SelfAttention(nn.Module):
    """Self-attention mechanism for focusing on important features"""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply self-attention to hidden states"""
        # Calculate attention weights
        attention_weights = F.softmax(self.attention(hidden), dim=1)
        
        # Apply attention to hidden states
        attended = hidden * attention_weights
        
        return attended, attention_weights
        
class NeuroEvolutionOptimizer:
    """Optimizer using neuroevolution for network improvement"""
    
    def __init__(
        self,
        population_size: int = 50,
        mutation_rate: float = 0.1,
        mutation_strength: float = 0.1
    ):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.population = []
        
    def initialize_population(self, network_config: NetworkConfig) -> None:
        """Initialize population of neural networks"""
        self.population = [
            NeuralProcessor(network_config)
            for _ in range(self.population_size)
        ]
        
  def evolve(self, fitness_func: callable) -> NeuralProcessor:
        """Evolve population and return best network"""
        # Evaluate fitness for each network
        fitness_scores = [
            fitness_func(network)
            for network in self.population
        ]
        
        # Sort population by fitness
        sorted_population = [
            x for _, x in sorted(
                zip(fitness_scores, self.population),
                key=lambda pair: pair[0],
                reverse=True
            )
        ]
        
        # Select top performers
        elite_size = self.population_size // 4
        elites = sorted_population[:elite_size]
        
        # Create new population
        new_population = elites.copy()
        
        # Fill rest of population with mutated versions of elites
        while len(new_population) < self.population_size:
            parent = random.choice(elites)
            child = self._mutate(parent)
            new_population.append(child)
            
        self.population = new_population
        return elites[0]  # Return best network
        
    def _mutate(self, network: NeuralProcessor) -> NeuralProcessor:
        """Create mutated copy of network"""
        # Create deep copy
        mutated = copy.deepcopy(network)
        
        # Mutate weights and biases
        with torch.no_grad():
            for param in mutated.parameters():
                if random.random() < self.mutation_rate:
                    noise = torch.randn_like(param) * self.mutation_strength
                    param.add_(noise)
                    
        return mutated

class AdaptiveController:
    """Controller for adapting neural network behavior"""
    
    def __init__(
        self,
        network: NeuralProcessor,
        learning_rate_bounds: Tuple[float, float] = (0.0001, 0.1),
        adaptation_rate: float = 0.01
    ):
        self.network = network
        self.lr_bounds = learning_rate_bounds
        self.adaptation_rate = adaptation_rate
        self.performance_history = []
        self.current_lr = learning_rate_bounds[1]
        
    def adapt(self, performance_metric: float) -> None:
        """Adapt network parameters based on performance"""
        self.performance_history.append(performance_metric)
        
        if len(self.performance_history) > 1:
            # Calculate performance change
            delta = self.performance_history[-1] - self.performance_history[-2]
            
            # Adjust learning rate
            if delta > 0:
                # Performance improving - increase learning rate
                self.current_lr *= (1 + self.adaptation_rate)
            else:
                # Performance degrading - decrease learning rate
                self.current_lr *= (1 - self.adaptation_rate)
                
            # Ensure learning rate stays within bounds
            self.current_lr = max(self.lr_bounds[0], min(self.current_lr, self.lr_bounds[1]))
            
    def get_adapted_parameters(self) -> Dict[str, float]:
        """Get current adapted parameters"""
        return {
            'learning_rate': self.current_lr,
            'recent_performance': self.performance_history[-1] if self.performance_history else None
        }

class MetaLearner:
    """Meta-learning system for optimizing neural architectures"""
    
    def __init__(
        self,
        base_config: NetworkConfig,
        max_layers: int = 5,
        max_neurons_per_layer: int = 256
    ):
        self.base_config = base_config
        self.max_layers = max_layers
        self.max_neurons = max_neurons_per_layer
        self.architecture_history = []
        
    def generate_architecture(self) -> NetworkConfig:
        """Generate new network architecture"""
        num_layers = random.randint(1, self.max_layers)
        hidden_sizes = [
            random.randint(self.base_config.input_size, self.max_neurons)
            for _ in range(num_layers)
        ]
        
        return NetworkConfig(
            input_size=self.base_config.input_size,
            hidden_sizes=hidden_sizes,
            output_size=self.base_config.output_size,
            activation=random.choice(['relu', 'tanh', 'leaky_relu']),
            dropout_rate=random.uniform(0.1, 0.5),
            use_batch_norm=random.choice([True, False])
        )
        
    def update_architecture_history(
        self,
        config: NetworkConfig,
        performance: float
    ) -> None:
        """Update history with architecture performance"""
        self.architecture_history.append({
            'config': config,
            'performance': performance
        })
        
    def get_best_architecture(self) -> NetworkConfig:
        """Get best performing architecture"""
        if not self.architecture_history:
            return self.base_config
            
        best = max(
            self.architecture_history,
            key=lambda x: x['performance']
        )
        return best['config']