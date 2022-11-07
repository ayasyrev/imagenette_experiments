# First version.



How to repeat results:  

All notebooks can be run on google colab.  

Install this repo version v0.0.1.:  
` pip install -e git+https://github.com/ayasyrev/imagenette_experiments@v0.0.1`  


Now import Model constructor and helper utils as:


```python
from imagenette_experiments.train_utils import *
from imagenette_experiments.trick_model import *
```

Now create model constructor:


```python
model = Model()
```

Now we can check model, for example:


```python
model.body.l_1.bl_0
```
???+ done "output"  
    <pre>NewResBlock(
      (reduce): MaxBlurPool2d()
      (convs): Sequential(
        (conv_0): ConvLayer(
          (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act_fn): Mish()
        )
        (conv_1): ConvLayer(
          (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act_fn): Mish()
        )
        (conv_2): ConvLayer(
          (conv): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (idconv): ConvLayer(
        (conv): Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (merge): Mish()
    )



Lets create Learner:


```python
learn = get_learn(woof=1, size=128, bs=64)
```
???+ done "output"  
    <pre>data path   /root/.fastai/data/imagewoof2
    Learn path /root/.fastai/data/imagewoof2


Now we cat train it regular fastai way, for example fit with annealing:

`learn.fit_fc(tot_epochs=5, lr=1e-4, moms=(0.95,0.95), start_pct=0.72)`
