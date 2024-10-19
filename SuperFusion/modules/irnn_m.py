import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class IRNN(nn.Module):
    def __init__(self, channels):
        super(IRNN, self).__init__()
        self.channels = channels
        
        # Weights and biases for each direction
        self.weight_up = nn.Parameter(torch.Tensor(channels, 1, 1))
        self.weight_right = nn.Parameter(torch.Tensor(channels, 1, 1))
        self.weight_down = nn.Parameter(torch.Tensor(channels, 1, 1))
        self.weight_left = nn.Parameter(torch.Tensor(channels, 1, 1))
        
        self.bias_up = nn.Parameter(torch.Tensor(channels))
        self.bias_right = nn.Parameter(torch.Tensor(channels))
        self.bias_down = nn.Parameter(torch.Tensor(channels))
        self.bias_left = nn.Parameter(torch.Tensor(channels))
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_up)
        nn.init.xavier_uniform_(self.weight_right)
        nn.init.xavier_uniform_(self.weight_down)
        nn.init.xavier_uniform_(self.weight_left)
        nn.init.zeros_(self.bias_up)
        nn.init.zeros_(self.bias_right)
        nn.init.zeros_(self.bias_down)
        nn.init.zeros_(self.bias_left)

    def forward(self, input_feature):
        batch_size, channels, height, width = input_feature.shape
        
        # Up direction
        output_up = torch.zeros_like(input_feature)
        for h in range(height - 1, -1, -1):
            if h == height - 1:
                output_up[:, :, h, :] = F.relu(input_feature[:, :, h, :])
            else:
                temp = F.conv2d(output_up[:, :, h+1:h+2, :], self.weight_up, self.bias_up, groups=channels)
                output_up[:, :, h, :] = F.relu(temp.squeeze(2) + input_feature[:, :, h, :])
        
        # Right direction
        output_right = torch.zeros_like(input_feature)
        for w in range(width):
            if w == 0:
                output_right[:, :, :, w] = F.relu(input_feature[:, :, :, w])
            else:
                temp = F.conv2d(output_right[:, :, :, w-1:w], self.weight_right, self.bias_right, groups=channels)
                output_right[:, :, :, w] = F.relu(temp.squeeze(3) + input_feature[:, :, :, w])
        
        # Down direction
        output_down = torch.zeros_like(input_feature)
        for h in range(height):
            if h == 0:
                output_down[:, :, h, :] = F.relu(input_feature[:, :, h, :])
            else:
                temp = F.conv2d(output_down[:, :, h-1:h, :], self.weight_down, self.bias_down, groups=channels)
                output_down[:, :, h, :] = F.relu(temp.squeeze(2) + input_feature[:, :, h, :])
        
        # Left direction
        output_left = torch.zeros_like(input_feature)
        for w in range(width - 1, -1, -1):
            if w == width - 1:
                output_left[:, :, :, w] = F.relu(input_feature[:, :, :, w])
            else:
                temp = F.conv2d(output_left[:, :, :, w+1:w+2], self.weight_left, self.bias_left, groups=channels)
                output_left[:, :, :, w] = F.relu(temp.squeeze(3) + input_feature[:, :, :, w])
        
        return output_up, output_right, output_down, output_left


class irnn(nn.Module):
    def __init__(self):
        super(irnn, self).__init__()
        self.weight_up = None
        self.weight_right = None
        self.weight_down = None
        self.weight_left = None
        self.bias_up = None
        self.bias_right = None
        self.bias_down = None
        self.bias_left = None

    def _init_weights(self, channels):
        self.weight_up = nn.Parameter(torch.Tensor(channels, 1, 1))
        self.weight_right = nn.Parameter(torch.Tensor(channels, 1, 1))
        self.weight_down = nn.Parameter(torch.Tensor(channels, 1, 1))
        self.weight_left = nn.Parameter(torch.Tensor(channels, 1, 1))
        self.bias_up = nn.Parameter(torch.Tensor(channels))
        self.bias_right = nn.Parameter(torch.Tensor(channels))
        self.bias_down = nn.Parameter(torch.Tensor(channels))
        self.bias_left = nn.Parameter(torch.Tensor(channels))
        self.reset_parameters()

    def reset_parameters(self):
        for weight in [self.weight_up, self.weight_right, self.weight_down, self.weight_left]:
            nn.init.xavier_uniform_(weight)
        for bias in [self.bias_up, self.bias_right, self.bias_down, self.bias_left]:
            nn.init.zeros_(bias)

    def forward(self, input_feature):
        batch_size, channels, height, width = input_feature.shape
        
        if self.weight_up is None:
            self._init_weights(channels)
        
        # Up direction
        output_up = torch.zeros_like(input_feature)
        for h in range(height - 1, -1, -1):
            if h == height - 1:
                output_up[:, :, h, :] = F.relu(input_feature[:, :, h, :])
            else:
                temp = F.conv1d(output_up[:, :, h+1, :].unsqueeze(2), self.weight_up, self.bias_up, groups=channels).squeeze(2)
                output_up[:, :, h, :] = F.relu(temp + input_feature[:, :, h, :])
        
        # Right direction
        output_right = torch.zeros_like(input_feature)
        for w in range(width):
            if w == 0:
                output_right[:, :, :, w] = F.relu(input_feature[:, :, :, w])
            else:
                temp = F.conv1d(output_right[:, :, :, w-1].unsqueeze(2), self.weight_right, self.bias_right, groups=channels).squeeze(2)
                output_right[:, :, :, w] = F.relu(temp + input_feature[:, :, :, w])
        
        # Down direction
        output_down = torch.zeros_like(input_feature)
        for h in range(height):
            if h == 0:
                output_down[:, :, h, :] = F.relu(input_feature[:, :, h, :])
            else:
                temp = F.conv1d(output_down[:, :, h-1, :].unsqueeze(2), self.weight_down, self.bias_down, groups=channels).squeeze(2)
                output_down[:, :, h, :] = F.relu(temp + input_feature[:, :, h, :])
        
        # Left direction
        output_left = torch.zeros_like(input_feature)
        for w in range(width - 1, -1, -1):
            if w == width - 1:
                output_left[:, :, :, w] = F.relu(input_feature[:, :, :, w])
            else:
                temp = F.conv1d(output_left[:, :, :, w+1].unsqueeze(2), self.weight_left, self.bias_left, groups=channels).squeeze(2)
                output_left[:, :, :, w] = F.relu(temp + input_feature[:, :, :, w])
        
        return output_up, output_right, output_down, output_left


# Usage example
if __name__ == "__main__":
    torch.manual_seed(0)  # for reproducibility
    batch_size, channels, height, width = 2, 16, 32, 32
    input_feature = torch.randn(batch_size, channels, height, width, requires_grad=True)
    # model = IRNN(channels)
    model = irnn(channels)
    output_up, output_right, output_down, output_left = model(input_feature)
    
    # Compute loss and gradients
    loss = output_up.sum() + output_right.sum() + output_down.sum() + output_left.sum()
    loss.backward()
    
    print("Forward pass completed successfully")
    print("Backward pass completed successfully")
    print("Input gradient shape:", input_feature.grad.shape)
    print("Weight up gradient shape:", model.weight_up.grad.shape)