import torch
from torch import nn
from math import pi
import numpy as np

class Embedding(nn.Module):
    def __init__(self, in_channels, N_freqs, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels 
        """
        super(Embedding, self).__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = in_channels*(len(self.funcs)*N_freqs+1)
        
        # self.B = np.random.randn(N_freqs,2)
        
        if logscale:
            self.freq_bands = 2**torch.linspace(0, N_freqs-1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2**(N_freqs-1), N_freqs)

    def forward(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...) 
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12

        Inputs:
            x: (B, self.in_channels)

        Outputs:
            out: (B, self.out_channels)
        """
        out = [x]
        
        # for freq_id in range(self.N_freqs):
        #     for func_id in range(len(self.funcs)):
        #         func = self.funcs[func_id]
        #         scale_cur = 2.3*self.B[freq_id, func_id]
        #         out += [func(2*pi*scale_cur*x)]
        
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq*x)]
 
        return torch.cat(out, -1)


class IdentityEmbedding(nn.Module):
    def __init__(self):
        super(IdentityEmbedding, self).__init__()

    def forward(self, x):
        return x



class NeRF(nn.Module):
    def __init__(self, D=8, W=256, in_channels_xy=82, skips=[]):
        """
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        in_channels_xy: number of input channels for xy (2+2*20*2=82 by default)
        skips: add skip connection in the Dth layer
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.in_channels_xy = in_channels_xy
        self.skips = skips

        # xy encoding layers
        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channels_xy, W)
            elif i in skips:
                layer = nn.Linear(W+in_channels_xy, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"uvxy_encoding_{i+1}", layer)
            
        self.uvxy_encoding_final1 = nn.Linear(W, W)

        self.uvxy_encoding_final2 = nn.Sequential(
                                nn.Linear(W, W//2),
                                nn.ReLU(True))
        # output layers
        self.rgb = nn.Sequential(
                        nn.Linear(W//2, 1),
                        nn.Sigmoid())

    def forward(self, x):
        """
        Inputs:
            x: (B, self.in_channels_xy) the embedded vector of position
            
        Outputs:
            out: (B, 1), gray value
        """
        input_uvxy = x

        uvxy_ = input_uvxy
        for i in range(self.D):
            if i in self.skips:
                uvxy_ = torch.cat([input_uvxy, uvxy_], -1)
            uvxy_ = getattr(self, f"uvxy_encoding_{i+1}")(uvxy_)

        uvxy_encoding_final1 = self.uvxy_encoding_final1(uvxy_)
        uvxy_encoding_final2 = self.uvxy_encoding_final2(uvxy_encoding_final1)
        out = self.rgb(uvxy_encoding_final2)

        return out



class MLP(nn.Module):
    def __init__(self, D=8, W=256, in_channels_xy=82, skips=[]):
        super(MLP, self).__init__()
        self.D = D
        self.W = W
        self.in_channels_xy = in_channels_xy
        self.skips = skips

        # xy encoding layers
        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channels_xy, W)
            elif i in skips:
                layer = nn.Linear(W+in_channels_xy, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"uvxy_encoding_{i+1}", layer)
            
        self.uvxy_encoding_final1 = nn.Linear(W, W)
        self.uvxy_encoding_final2 = nn.Sequential(
                                nn.Linear(W, W//2),
                                nn.ReLU(True))
        # output layers
        self.rgb = nn.Sequential(
                        nn.Linear(W//2, 1),
                        nn.Sigmoid())

    def forward(self, x):
        """
        Inputs:
            x: (B, self.in_channels_xy) the embedded vector of position
            
        Outputs:
            out: (B, 1), gray value
        """
        input_uvxy = x

        uvxy_ = input_uvxy
        for i in range(self.D):
            if i in self.skips:
                uvxy_ = torch.cat([input_uvxy, uvxy_], -1)
            uvxy_ = getattr(self, f"uvxy_encoding_{i+1}")(uvxy_)

        uvxy_encoding_final1 = self.uvxy_encoding_final1(uvxy_)
        uvxy_encoding_final2 = self.uvxy_encoding_final2(uvxy_encoding_final1)
        out = self.rgb(uvxy_encoding_final2)

        return out



# SRIREN Network
class SineLayer(nn.Module):
    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the nonlinearity.
    # Different signals may require different omega_0 in the first layer - this is a hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
    
    def forward_with_intermediate(self, input): 
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate
    
class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        # self.net.append(nn.Sigmoid())
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output, coords        

    def forward_with_activations(self, coords, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!'''
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)
                
                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()
                    
                activations['_'.join((str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else: 
                x = layer(x)
                
                if retain_grad:
                    x.retain_grad()
                    
            activations['_'.join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations
