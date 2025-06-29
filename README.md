# 🧠 CartPole DQN Agent

![CartPole Screenshot](https://github.com/user-attachments/assets/f369e5ed-29c8-4892-b92b-53f84badbe7a)

A Deep Q-Network (DQN) implementation using **PyTorch** to solve the classic **CartPole-v1** balancing problem from OpenAI Gym. The agent learns to balance a pole on a cart by taking smart left/right actions based on observed environment states.

---

## 🚀 What This Project Does

This agent learns to balance a pole by:

- Observing the environment (cart position, velocity, pole angle, angular velocity)
- Predicting the best action using a neural network
- Receiving rewards and learning from them
- Updating its strategy through Deep Q-Learning

Eventually, the agent balances the pole **smoothly and efficiently**, even across the full width of the environment.

---

## ✨ Features

- ✅ Deep Q-Network (DQN) built with PyTorch  
- ✅ Experience Replay Buffer  
- ✅ Epsilon-Greedy Exploration Strategy  
- ✅ Target Network Updates  
- ✅ Live Agent Visualization using Pygame  
- ✅ Automatic Checkpointing & Resume Support  
- ✅ Trained on OpenAI Gym’s CartPole-v1 environment

---

## 📚 What I Learned

This project helped me understand and apply key Reinforcement Learning concepts:

- How agents interact with environments via actions and rewards  
- The mechanics of Deep Q-Learning  
- Exploration vs Exploitation (Epsilon-Greedy Policy)  
- Building a memory buffer and training loop  
- Optimizing models using PyTorch  
- Visualizing real-time agent behavior

---

## ⚙️ How It Works

### 🧠 Environment: CartPole-v1
The environment provides a 4D state:
- Cart Position  
- Cart Velocity  
- Pole Angle  
- Pole Angular Velocity

### 🎮 Actions
- `0` → Move Cart Left  
- `1` → Move Cart Right

### 🏆 Reward
- +1 reward per timestep the pole remains upright  
- Episode ends if the pole falls or the cart goes off-screen

---

## 🧪 Training Summary

The agent is trained over 400 episodes. Initially, it explores randomly. Around episode 150, it begins balancing the pole intelligently. It eventually achieves a **maximum reward of 315**.

- Model checkpoints are saved after every episode in the `checkpoints/` directory.
- If training is interrupted, it **automatically resumes from the last saved episode**.

---

## 📁 Project Structure

- `cartpole.py` – Core DQN agent, environment handling, training logic  
- `GUI.py` – Pygame-based visual interface to view agent behavior live  
- `run_trained_model.py` – Runs the agent using saved model weights  
- `requirements.txt` – All project dependencies

---

## 🛠️ Setup & Installation

Make sure Python 3.x is installed. Then install all required libraries using:

```bash
pip install -r requirements.txt
```

This will automatically install all necessary packages including:
- `torch`
- `gym`
- `numpy`
- `pygame`

---

## ▶️ Usage

To **train the agent**, run:

```bash
python cartpole.py
```

To **run the trained agent**, use:

```bash
python run_trained_model.py
```

---

## 💬 Feedback

If you have suggestions or run into issues, feel free to open an issue or reach out. I'm always looking to learn and improve this project!

---
