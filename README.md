# CartPole DQN Agent

![Screenshot 2025-05-24 145258](https://github.com/user-attachments/assets/f369e5ed-29c8-4892-b92b-53f84badbe7a)


A Deep Q-Network (DQN) implementation using PyTorch to solve the classic CartPole-v1 balancing problem from OpenAI Gym. The agent learns to balance a pole on a cart by taking left or right actions based on observed states. Over time, it improves its behavior and learns how to maximize the duration of balance by making smarter decisions.

---

## 🧠 What This Project Does

This project trains an artificial agent to **balance a pole** on a cart by:

- Observing the current environment state (cart position, velocity, pole angle, etc.).
- Predicting the best action (move left or right).
- Receiving a reward after each step and learning from it.
- Adjusting its strategy (policy) using **Deep Q-Learning**.

Eventually, the agent learns to move the cart **smoothly left and right** to keep the pole upright — even across the full width of the environment.

---

## 🚀 Features

- ✅ Deep Q-Network using PyTorch  
- ✅ Experience Replay Buffer  
- ✅ Epsilon-Greedy Exploration  
- ✅ Target Network Updates  
- ✅ Trained on CartPole-v1 environment from OpenAI Gym  
- ✅ Live rendering for visualization  

---

## 📚 What I Learned

This project helped me understand and apply several key concepts in Reinforcement Learning and Deep Learning:

- 🔁 How an agent interacts with an environment using actions and rewards
- 🧠 How Deep Q-Learning works under the hood
- 💡 Exploration vs Exploitation (Epsilon-Greedy Policy)
- 🧮 Backpropagation, optimization, and loss minimization in PyTorch
- 🧱 Building a training loop with memory buffer and target updates
- 🎮 Real-time agent behavior visualization

It was really cool to see the agent go from failing early to **balancing the pole for long durations** just by learning from rewards and its own past experiences.

---

## 🛠️ How It Works

### 📦 Environment
- **CartPole-v1** has 4 key state values:
  - Cart position
  - Cart velocity
  - Pole angle
  - Pole angular velocity

### 🕹️ Actions
- Move Cart Left (`0`)
- Move Cart Right (`1`)

### 🏆 Reward
- +1 for every timestep the pole stays upright  
- Episode ends if the pole falls or the cart goes off-screen

---

## 🧪 Training Overview

The agent is trained over multiple episodes. At first, it performs randomly (exploring), but gradually starts **exploiting its learned Q-values** to act better. After ~150 episodes, the agent begins to show signs of intelligence — **balancing longer, adjusting faster, and moving efficiently**.

---

## 🧰 Requirements

- Python 3.x  
- PyTorch  
- NumPy  
- OpenAI Gym  

Install dependencies:
```bash
pip install torch gym numpy
