
# DESCRIPTION

a `class` wrap the perceptual loss to quickly compare the similarity between two image 

# INTRODUCTION

| perceptual loss | similarity |
|-----------------|------------|
| 0               | high       |


# USE

~~~
# debug.py

import torch
from perceptual_loss_wrapper import PerceptualLossWrapper

perceptual_loss = PerceptualLossWrapper('lpips')

images_a = torch.randn([4, 3, 32, 32])
images_b = torch.randn([4, 3, 32, 32])
images_c = torch.rand([4, 3, 32, 32])


similarity = perceptual_loss(images_a, images_b)
>> tensor([[[[0.2784]]],
        [[[0.3096]]],
        [[[0.2768]]],
        [[[0.2634]]]], grad_fn=<AddBackward0>)

similarity = perceptual_loss(images_a, images_c)
>> tensor([[[[0.3473]]],
        [[[0.4045]]],
        [[[0.3761]]],
        [[[0.3826]]]], grad_fn=<AddBackward0>)
~~~

# TODO

- [ ] refactor
- [ ] beautify the format
- [ ] check import
- [ ] support cpu

# ACKNOWLEDGE

overwritten on the code from [ wpeebles /
gangealing ](https://github.com/wpeebles/gangealing)
