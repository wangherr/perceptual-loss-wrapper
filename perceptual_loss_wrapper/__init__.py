from perceptual_loss_wrapper.lpips import get_perceptual_loss


class PerceptualLossWrapper:
    def __init__(self, name='lpips'):
        assert name in ['vgg_ssl', 'lpips']
        self.loss_fn = get_perceptual_loss(name)

    def __call__(self, input_a, input_b):
        return self.loss_fn(input_a, input_b)
