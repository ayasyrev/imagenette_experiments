# imagenette_experiments
> experiments with fastai imagenette / imagewoof datasets


Notebooks with train results.

This results was done with experimental model - XResnet with modification. I use pool layer plus convolution stride 1 instead of convolution stride 2.  
Explanatiom about model here: https://github.com/ayasyrev/imagenette_experiments/blob/master/ResnetTrick_create_model_fit.ipynb

Current results vs results on leaderboard - ImageWOOF:

| Size (px) | Epochs |   | Accuracy | # Runs | My res | URL |  Comments |
|--|--|--|--|--|--|--| -- |
|128|5|  |73.37%|5, mean| |
|128|20||85.52%|5, mean|86.10% | |
|128|80||87.20%|1| 87.60% | | --
|128|200||87.20%|1| | |- training
||||||| 
|192|5||75.94%|5, mean| 77.87% | [link](https://github.com/ayasyrev/imagenette_experiments/blob/master/Woof_MaxBlurPool_ResnetTrick_s192bs32.ipynb) | added to board
|192|20||87.25%|5, mean| 87.85% | [link](https://github.com/ayasyrev/imagenette_experiments/blob/master/Woof_MaxBlurPool_ResnetTrick_s192bs32.ipynb)  | added to board
|192|80||89.21%|1| 89.69% | | --
|192|200||89.54%|1| 90.35% | | --
|||||||
|256|5||76.87%|5, mean| 78,84% | [link](https://github.com/ayasyrev/imagenette_experiments/blob/master/Woof_MaxBlurPool_ResnetTrick_s256bs32.ipynb)| added to board
|256|20||88.29%|5, mean| 88,58% | [link](https://github.com/ayasyrev/imagenette_experiments/blob/master/Woof_MaxBlurPool_ResnetTrick_s256bs32.ipynb)| added to board
|256|80||90.48%|1| 90.65% | | --
|256|200||90.38%|1| 91.14% | | --



How to repeat results:  
All notebooks can be run on google colab.  

Install this repo:
` pip install -e git+https://github.com/ayasyrev/imagenette_experiments`  
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




    NewResBlock(
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

    data path   /root/.fastai/data/imagewoof2
    Learn path /root/.fastai/data/imagewoof2


Now we cat train it regular fastai way, for example fit with annealing:

`learn.fit_fc(tot_epochs=5, lr=1e-4, moms=(0.95,0.95), start_pct=0.72)`
