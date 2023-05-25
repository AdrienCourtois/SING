# Official implementation of SING : A Plug-and-Play DNN Training Technique

Usage:
```python
import torch
from sing import SING

optimizer = SING(model.parameters(), lr=1e-1, weight_decay=5e-3)

for epoch in range(1, epochs + 1):
  for x, y in train_loader:
    loss = criterion(model(x), y)
    loss.backward()
    
    optimizer.step()
    optimizer.zero_grad()
```

The `SING` class extends the `torch.optim.Optimizer` class and can therefore be used as any PyTorch optimizer.

## Hyper-parameters
The learning rate of `SING` must be set to the highest value possible before explosing. Usually, the best learning rate for `SING` is 10 times the best learning rate for `AdamW`.
A good search space is &lcub; $5 \times 10^{-3}, 1 \times 10^{-2}, 5 \times 10^{-2}, 1 \times 10^{-1}$ &rcub;.

The weight decay is an important hyper-parameter, we advise practionners to tune it. Usually, the best weight decay for `SING` is 10 times lower than the best weight decay for `AdamW`.
A good search space is &lcub; $5 \times 10^{-4}, 5 \times 10^{-3}, 5 \times 10^{-2}, 5 \times 10^{-1}$ &rcub;.

## Learning rate scheduler
It is recommanded to use one of these two techniques:
- Linear warmup for 5% of the iterations, followed by a cosine decay for the remaining iterations.
- Linear warmup for 5% of the iterations, followed by a constant learning rate for 60% of the iterations and a cosine decay for the remaining iterations.

## Further recommandations
`SING` can be difficult to use with some operations affecting the channels individually. Such operations include LayerNorm [1], LayerScale [2].
We advise practionners to track the evolution of the gradient norm throughout training. If the training loss decreases while the gradient norm increases, the learning of these layers should be disabled. For the LayerNorm, it amounts to do:
```python
for module in model.modules():
  if isinstance(module, nn.LayerNorm):
    module.weight.requires_grad_(False)
    module.bias.requires_grad_(False)
```
Note the normalization is still applied and only the rescaling is affected. Doing so is not detrimental and is actually linked to better generalization [3].

## References
[1] Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). Layer normalization. *arXiv preprint arXiv:1607.06450*. \
[2] Touvron, H., Cord, M., Sablayrolles, A., Synnaeve, G., & Jégou, H. (2021). *Going deeper with image transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision* (pp. 32-42). \
[3] Xu, J., Sun, X., Zhang, Z., Zhao, G., & Lin, J. (2019). Understanding and improving layer normalization. *Advances in Neural Information Processing Systems, 32*.
