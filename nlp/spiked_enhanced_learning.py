"""
Spiked Enhanced Learning for NLP
Kết hợp SNN và Reinforcement Learning để học ngữ pháp động
"""

import torch
import torch.nn as nn
import torch.optim as optim
import snntorch as snn
from snntorch import spikegen, surrogate
import numpy as np
from collections import deque
import random

class SpikedEnhancedLearner(nn.Module):
    """
    Spiking Neural Network with Enhanced Learning
    Dùng spike-timing dependent plasticity (STDP) và policy gradient
    """
    
    def __init__(self, input_dim=256, hidden_dim=512, output_dim=10, num_timesteps=25):
        super().__init__()
        
        self.num_timesteps = num_timesteps
        self.hidden_dim = hidden_dim
        
        # SNN layers with LIF neurons
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.lif1 = snn.Leaky(beta=0.9, spike_grad=surrogate.fast_sigmoid())
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.lif2 = snn.Leaky(beta=0.9, spike_grad=surrogate.fast_sigmoid())
        
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.lif3 = snn.Leaky(beta=0.9, spike_grad=surrogate.fast_sigmoid())
        
        # Policy network (actor)
        self.actor = nn.Sequential(
            nn.Linear(output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        
        # Value network (critic)
        self.critic = nn.Sequential(
            nn.Linear(output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # STDP parameters
        self.std_decay = 0.99
        self.std_learning_rate = 0.01
        
        # Memory for experience replay
        self.memory = deque(maxlen=10000)
        
    def forward(self, x):
        """
        Forward pass through SNN
        x: [batch, input_dim]
        """
        batch_size = x.shape[0]
        
        # Initialize membrane potentials
        mem1 = self.lif1.init_leaky(batch_size, self.hidden_dim, device=x.device)
        mem2 = self.lif2.init_leaky(batch_size, self.hidden_dim, device=x.device)
        mem3 = self.lif3.init_leaky(batch_size, self.hidden_dim, device=x.device)
        
        spike_records = []
        
        for t in range(self.num_timesteps):
            # Time-varying input (simulate spike patterns)
            x_t = x * (0.95 ** t)  # Exponential decay
            
            # Layer 1
            cur1 = self.fc1(x_t)
            spk1, mem1 = self.lif1(cur1, mem1)
            
            # Layer 2
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            
            # Layer 3
            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)
            
            spike_records.append(spk3)
        
        # Aggregate spikes over time
        spikes = torch.stack(spike_records).sum(dim=0) / self.num_timesteps
        
        # Policy and value
        policy_logits = self.actor(spikes)
        value = self.critic(spikes)
        
        return policy_logits, value, spikes
    
    def get_action(self, state, epsilon=0.1):
        """Chọn action dựa trên policy"""
        with torch.no_grad():
            policy_logits, value, spikes = self.forward(state)
            probs = torch.softmax(policy_logits, dim=-1)
            
            if random.random() < epsilon:
                action = random.randint(0, policy_logits.shape[-1] - 1)
            else:
                action = probs.argmax().item()
            
            return action, probs[0, action].item(), value.item()
    
    def remember(self, state, action, reward, next_state, done):
        """Lưu experience vào memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def update_stdp(self, pre_spikes, post_spikes):
        """
        Spike-Timing Dependent Plasticity update
        Hebbian learning rule
        """
        # Simplified STDP: pre-before-post strengthens, post-before-pre weakens
        for i in range(len(pre_spikes)):
            for j in range(len(post_spikes)):
                delta_t = pre_spikes[i] - post_spikes[j]
                if abs(delta_t) < 10:  # Time window
                    if delta_t > 0:  # Pre before post
                        weight_change = self.std_learning_rate * np.exp(-delta_t / 5)
                    else:  # Post before pre
                        weight_change = -self.std_learning_rate * np.exp(delta_t / 5)
                    
                    # Apply weight change (simplified)
                    self.fc1.weight.data[i, j] += weight_change
    
    def learn_from_replay(self, batch_size=32, gamma=0.99):
        """Experience replay với policy gradient"""
        if len(self.memory) < batch_size:
            return 0
        
        batch = random.sample(self.memory, batch_size)
        
        total_loss = 0
        for state, action, reward, next_state, done in batch:
            # Get policy and value
            policy_logits, value, _ = self.forward(state)
            _, next_value, _ = self.forward(next_state) if next_state is not None else (None, 0, None)
            
            # Compute advantage
            target = reward + (1 - done) * gamma * next_value
            advantage = target - value
            
            # Policy loss (actor)
            probs = torch.softmax(policy_logits, dim=-1)
            log_prob = torch.log(probs[0, action] + 1e-8)
            policy_loss = -log_prob * advantage
            
            # Value loss (critic)
            value_loss = advantage ** 2
            
            # Total loss
            loss = policy_loss + 0.5 * value_loss
            total_loss += loss.item()
        
        return total_loss / len(batch)


class GrammarAdaptor:
    """Điều chỉnh ngữ pháp động dựa trên Spiked Enhanced Learning"""
    
    def __init__(self):
        self.learner = SpikedEnhancedLearner()
        self.optimizer = optim.Adam(self.learner.parameters(), lr=0.001)
        self.grammar_rules = self.init_grammar_rules()
    
    def init_grammar_rules(self):
        """Khởi tạo các quy tắc ngữ pháp tiếng Việt"""
        return {
            'S -> NP VP': 1.0,
            'NP -> N | Det N | Adj N': 1.0,
            'VP -> V | V NP | V NP PP': 1.0,
            'PP -> P NP': 1.0,
            'S -> S Conj S': 0.5
        }
    
    def adapt_grammar(self, sentence, reward):
        """Điều chỉnh ngữ pháp dựa trên phản hồi"""
        # Encode sentence thành state
        state = self.encode_sentence(sentence)
        
        # Get action từ policy
        action, prob, value = self.learner.get_action(state)
        
        # Apply grammar rule based on action
        rule_idx = action % len(self.grammar_rules)
        rule = list(self.grammar_rules.keys())[rule_idx]
        
        # Update rule weight based on reward
        if reward > 0:
            self.grammar_rules[rule] = min(1.0, self.grammar_rules[rule] + 0.05)
        else:
            self.grammar_rules[rule] = max(0.0, self.grammar_rules[rule] - 0.05)
        
        return rule
    
    def encode_sentence(self, sentence):
        """Encode sentence thành tensor state"""
        # Simplified encoding
        words = sentence.split()
        features = torch.randn(1, 256)  # Placeholder
        return features


# Test
if __name__ == "__main__":
    print("🧠 Testing Spiked Enhanced Learning...")
    
    learner = SpikedEnhancedLearner()
    adaptor = GrammarAdaptor()
    
    # Test forward pass
    test_input = torch.randn(1, 256)
    policy, value, spikes = learner(test_input)
    
    print(f"✅ Policy shape: {policy.shape}")
    print(f"✅ Value: {value.item():.4f}")
    print(f"✅ Spike rate: {spikes.mean().item():.4f}")
    
    # Test grammar adaptation
    sentence = "Robot học ngữ pháp tiếng Việt"
    rule = adaptor.adapt_grammar(sentence, reward=1.0)
    print(f"✅ Adapted grammar rule: {rule}")