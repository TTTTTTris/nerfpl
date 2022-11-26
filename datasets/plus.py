# sin
import numpy as np
import torch
from torch.autograd import Function
class sin_plus(Function):
  @staticmethod    
  def forward(ctx, input):    
    ctx.save_for_backward(input)    
    return input.sin()

  @staticmethod
  def backward(ctx, grad_output):
    input, = ctx.saved_tensors
    grad_x = grad_output.clone()
    grad_x = grad_output*torch.cos(input)
    grad_x[abs(torch.sin(input)) >= 0.999] = 0.0447*torch.sign(grad_x[abs(torch.sin(input)) >= 0.999])
    # grad_x[abs(torch.sin(input)) >= 0.9] = 0.1*torch.sign(grad_x[abs(torch.sin(input)) >= 0.9])
    return grad_x

# cos
class cos_plus(Function):
  @staticmethod    
  def forward(ctx, input):    
    ctx.save_for_backward(input)    
    return input.cos()

  @staticmethod
  def backward(ctx, grad_output):
    input, = ctx.saved_tensors
    grad_x = grad_output.clone()
    grad_x = -grad_output*torch.sin(input)
    grad_x[abs(torch.cos(input)) >= 0.999] = 0.0447*torch.sign(grad_x[abs(torch.cos(input)) >= 0.999])
    # grad_x[abs(torch.cos(input)) >= 0.9] = 0.1*torch.sign(grad_x[abs(torch.cos(input)) >= 0.9])
    return grad_x
