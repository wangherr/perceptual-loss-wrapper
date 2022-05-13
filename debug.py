import torch
from perceptual_loss_wrapper import PerceptualLossWrapper

perceptual_loss = PerceptualLossWrapper('lpips')

images_a = torch.randn([4, 3, 32, 32])
images_b = torch.randn([4, 3, 32, 32])
images_c = torch.rand([4, 3, 32, 32])

similarity = perceptual_loss(images_a, images_b)
print(similarity)

similarity = perceptual_loss(images_a, images_c)
print(similarity)
