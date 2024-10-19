import torch
import torch.autograd as autograd
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def div_loss(D, real_x, fake_x, wp: int = 6):
    alpha = torch.rand((real_x.shape[0], 1, 1, 1)).to(device)
    x_ = (alpha * real_x + (1 - alpha) * fake_x).requires_grad_(True)
    y_ = D(x_)
    grad = autograd.grad(
        outputs=y_,
        inputs=x_,
        grad_outputs=torch.ones_like(y_),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grad = grad.view(x_.shape[0], -1)
    div = (grad.norm(2, dim=1) ** wp).mean()
    return div
