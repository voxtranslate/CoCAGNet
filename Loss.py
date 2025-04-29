# loss.py
import lpips
from torchvision.models import vgg16, VGG16_Weights
from pytorch_msssim import SSIM, MS_SSIM
from torch.autograd import Variable

class TotalVariationLoss(nn.Module):
    def __init__(self):
        super(TotalVariationLoss, self).__init__()

    def forward(self, x):
        # Calculate gradients in x and y directions
        h_tv = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        w_tv = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])

        # Return sum of both gradients
        return torch.mean(h_tv) + torch.mean(w_tv)

class CharbonnierLoss(nn.Module):
    def __init__(self, epsilon=1e-3):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred, target):
        diff = pred - target
        loss = torch.mean(Variable(torch.sqrt(diff * diff + self.epsilon * self.epsilon).type(torch.FloatTensor), requires_grad=True))
        return loss

class MSEGDL(nn.Module):
    def __init__(self, lambda_mse=1, lambda_gdl=1):
        super(MSEGDL, self).__init__()
        self.lambda_mse = lambda_mse
        self.lambda_gdl = lambda_gdl

    def forward(self, inputs, targets):

        squared_error = (inputs - targets).pow(2)
        gradient_diff_i = (inputs.diff(axis=-1)-targets.diff(axis=-1)).pow(2)
        gradient_diff_j =  (inputs.diff(axis=-2)-targets.diff(axis=-2)).pow(2)
        loss = (self.lambda_mse*squared_error.sum() + self.lambda_gdl*gradient_diff_i.sum() + self.lambda_gdl*gradient_diff_j.sum())/inputs.numel()

        return loss

class SSIMLoss(nn.Module):
    def __init__(self, data_range=1.0, size_average=True):
        super(SSIMLoss, self).__init__()
        self.data_range = data_range
        self.size_average = size_average

    def forward(self, img1, img2):
        return 1 - Variable(ssim(img1, img2, data_range=self.data_range, size_average=self.size_average).type(torch.FloatTensor), requires_grad=True)

class MSSSIMLoss(nn.Module):
    def __init__(self, data_range=1.0, size_average=True):
        super(MSSSIMLoss, self).__init__()
        self.data_range = data_range
        self.size_average = size_average

    def forward(self, img1, img2):
        return 1 - Variable(ms_ssim(img1, img2, data_range=self.data_range, size_average=self.size_average).type(torch.FloatTensor), requires_grad=True)


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.lpips = lpips.LPIPS(net='vgg')
        self.lpips.eval()
        self.lpips.requires_grad_(False)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.lpips(input, target).mean()

class VGGLoss(nn.Module):
    def __init__(self, layer=36):
        super().__init__()

        self.vgg = vgg19(weights=VGG19_Weights.DEFAULT).features[:layer].eval()
        self.loss = nn.MSELoss()

        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, output, target):
        self.vgg.eval()
        vgg_input_features = self.vgg(output)
        vgg_target_features = self.vgg(target)
        loss = self.loss(vgg_input_features, vgg_target_features)
        del vgg_input_features, vgg_target_features
        gc.collect()
        torch.cuda.empty_cache()
        return loss

# Combined Loss Function
class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.vgg_loss = VGGLoss()
        self.gan_loss = MSEGDL()
        self.ms_loss  = SSIMLoss()
        self.l1_loss  = CharbonnierLoss()

    def forward(self, output, target):
        vgg_loss = self.vgg_loss(output, target)
        gan_loss = self.gan_loss(output, target)
        ms_loss  = self.ms_loss(output, target)
        l1_loss  = self.l1_loss(output, target)
        return 10 * l1_loss + 8 * vgg_loss + 6 * ms_loss + 3 * gan_loss

def gradient_penalty(critic, real, fake, device):
    batch_size, channels, height, width = real.shape
    alpha = torch.rand((batch_size, 1, 1, 1)).repeat(1, channels, height, width).to(device)
    interpolated_images = real * alpha + fake.detach() * (1 - alpha)
    interpolated_images.requires_grad_(True)

    mixed_scores = critic(interpolated_images)

    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)

    return gradient_penalty