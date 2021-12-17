import numpy as np
import torch
import torch.nn.functional as F


def perform_qlearning_step(policy_net, target_net, optimizer, replay_buffer, batch_size, gamma, device, use_doubleqlearning = False):
    """ Perform a deep Q-learning step
    Parameters
    -------
    policy_net: torch.nn.Module
        policy Q-network
    target_net: torch.nn.Module
        target Q-network
    optimizer: torch.optim.Adam
        optimizer
    replay_buffer: ReplayBuffer
        replay memory storing transitions
    batch_size: int
        size of batch to sample from replay memory 
    gamma: float
        discount factor used in Q-learning update
    device: torch.device
        device on which to the models are allocated
    Returns
    -------
    float
        loss value for current learning step
    """

    # TODO: Run single Q-learning step
    """ Steps: 
        1. Sample transitions from replay_buffer
        2. Compute Q(s_t, a)
        3. Compute \max_a Q(s_{t+1}, a) for all next states.
        4. Mask next state values where episodes have terminated
        5. Compute the target
        6. Compute the loss
        7. Calculate the gradients
        8. Clip the gradients
        9. Optimize the model
    """

    # Tip: You can use use_doubleqlearning to switch the learning modality.

    samples = replay_buffer.sample(batch_size)
    obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = samples
    
    obs_batch = torch.tensor(obs_batch, dtype=torch.float).to(device)
    act_batch = torch.tensor(act_batch).to(device)
    rew_batch = torch.tensor(rew_batch, dtype=torch.float).to(device)
    next_obs_batch = torch.tensor(next_obs_batch, dtype=torch.float).to(device)
    done_mask = torch.tensor(done_mask, dtype=torch.float).to(device)

    q_values = policy_net(obs_batch)
    predicted_q_values = q_values[np.arange(0, len(q_values)), act_batch]

    with torch.no_grad():
        if use_doubleqlearning:
            target_q_values = target_net(next_obs_batch)
            policy_best_actions = policy_net(next_obs_batch).argmax(dim=1)[0]
            target_q_values = target_q_values[torch.arange(0, len(target_q_values)), policy_best_actions]
        else:
            target_q_values = target_net(next_obs_batch).max(dim=1)[0]
    target_q_values = rew_batch + gamma * target_q_values

    loss = (((target_q_values - predicted_q_values) * (1-done_mask) )**2).sum()

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return loss.item()


def update_target_net(policy_net, target_net):
    """ Update the target network
    Parameters
    -------
    policy_net: torch.nn.Module
        policy Q-network
    target_net: torch.nn.Module
        target Q-network
    """

    # TODO: Update target network
    target_net.load_state_dict(policy_net.state_dict())
