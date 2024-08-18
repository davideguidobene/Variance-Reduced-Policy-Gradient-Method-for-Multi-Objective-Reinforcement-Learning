import torch
import numpy as np

def discount_rewards(rewards, gamma):
    discounted_rewards = torch.empty_like(rewards, dtype=torch.float32)
    R = 0
    for t in reversed(range(0, len(rewards))):
        R = R * gamma + rewards[t]
        discounted_rewards[t] = R * (gamma**t)
    return discounted_rewards

def discounted_sum(rewards, gamma):  # rewards is list of np-arrays for deepsea
    J = np.zeros_like(rewards[0], dtype=np.float64)
    for t in range(len(rewards)):
        J += rewards[t] * (gamma**t)
    return J 


def get_grad(model):
    """
    Get the gradients of all parameters in the model as a dictionary of 1D tensors.
    
    Args:
    model (torch.nn.Module): The neural network model.
    
    Returns:
    dict: A dictionary where keys are parameter names and values are 1D gradient tensors.
    """
    grads = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads[name] = param.grad.view(-1)
        else:
            grads[name] = torch.zeros_like(param).view(-1)
    return grads

def set_grad(model, grad_dict):
    """
    Overwrite the gradients of all parameters in the model with provided dictionary of 1D tensors.
    
    Args:
    model (torch.nn.Module): The neural network model.
    grad_dict (dict): A dictionary where keys are parameter names and values are 1D gradient tensors.
    """
    for name, param in model.named_parameters():
        if name in grad_dict:
            param.grad = grad_dict[name].view(param.shape)

def grads_to_1d_tensor(grad_dict):
    """
    Convert a dictionary of 1D gradient tensors to a single 1D tensor.
    
    Args:
    grad_dict (dict): A dictionary where keys are parameter names and values are 1D gradient tensors.
    
    Returns:
    torch.Tensor: A single 1D tensor containing all gradients.
    """
    grads_list = [grad for grad in grad_dict.values()]
    grads_1d = torch.cat(grads_list)
    return grads_1d
