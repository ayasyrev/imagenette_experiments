# imagenette_experiments
> experiments with fastai imagenette / imagewoof datasets

This repo for store results of experiments with Imagenette / Imagewoof datasets.  

First "BATCH" of experiments stores at folder:  [Imagenette_Nbs_1](https://github.com/ayasyrev/imagenette_experiments/tree/master/Imagenette_Nbs_1)  
Four notebooks (Names stored with "Woof") leaved at "root" as there is urls from forums to it.  
That experiments are with fastai v1.  

Experiment results vs results on leaderboard at publish day.  
ImageWOOF dataset:  

| Size (px) | Epochs |   | Accuracy | # Runs | My res | URL |  Comments |
|--|--|--|--|--|--|--| -- |
|128|5|  |73.37%|5, mean| |
|128|20||85.52%|5, mean|86.10% | |
|128|80||87.20%|1| 87.63% |[notebook](https://github.com/ayasyrev/imagenette_experiments/blob/master/Imagenette_Nbs_1/Woof_MaxBlurPool_ResnetTrick_s128_e80_8763.ipynb) | 3 runs, start_pct=0.3
|128|200||87.20%|1|  88.30%| [notebook](https://github.com/ayasyrev/imagenette_experiments/blob/master/Imagenette_Nbs_1/Woof_MaxBlurPool_ResnetTrick_s128_e200_8830.ipynb) | 3 runs, start_pct=0.2
||||||| 
|192|5||75.94%|5, mean| 77.87% | [notebook](https://github.com/ayasyrev/imagenette_experiments/blob/master/Imagenette_Nbs_1/Woof_MaxBlurPool_ResnetTrick_s192bs32.ipynb) | added to board
|192|20||87.25%|5, mean| 87.85% | [notebook](https://github.com/ayasyrev/imagenette_experiments/blob/master/Imagenette_Nbs_1/Woof_MaxBlurPool_ResnetTrick_s192bs32.ipynb)  | added to board
|192|80||89.21%|1| 89.69% |[notebook](https://github.com/ayasyrev/imagenette_experiments/blob/master/Imagenette_Nbs_1/Woof_MaxBlurPool_ResnetTrick_s192bs32_e80_8969.ipynb) | 4 runs.
|192|200||89.54%|1| 90.35% |[notebook](https://github.com/ayasyrev/imagenette_experiments/blob/master/Imagenette_Nbs_1/Woof_MaxBlurPool_ResnetTrick_s192bs32_e200_9035.ipynb) | 2 runs.
|||||||
|256|5||76.87%|5, mean| 78,84% | [notebook](https://github.com/ayasyrev/imagenette_experiments/blob/master/Imagenette_Nbs_1/Woof_MaxBlurPool_ResnetTrick_s256bs32.ipynb)| added to board
|256|20||88.29%|5, mean| 88,58% | [notebook](https://github.com/ayasyrev/imagenette_experiments/blob/master/Imagenette_Nbs_1/Woof_MaxBlurPool_ResnetTrick_s256bs32.ipynb)| added to board
|256|80||90.48%|1| 90.63% | [notebook](https://github.com/ayasyrev/imagenette_experiments/blob/master/Imagenette_Nbs_1/Woof_MaxBlurPool_ResnetTrick_s256bs16_e80_9063.ipynb)| 2 runs, start_pct=0.4
|256|200||90.38%|1| 91.14% | [notebook](https://github.com/ayasyrev/imagenette_experiments/blob/master/Imagenette_Nbs_1/Woof_MaxBlurPool_ResnetTrick_s256bs16_e200_9114.ipynb)| 3 runs, start_pct=0.2



This results was done with experimental model - XResnet with modification.  
I used pool layer plus convolution stride 1 instead of convolution stride 2.  
And instead of regular pytorch pool (AveragePool2d and MaxPool2d) i used MaxBlurPool as described here:   
[fastai forum topic](https://forums.fast.ai/t/imagenette-imagewoof-leaderboards/45822/20?u=a_yasyrev)  
[github ducha-aiki](https://github.com/ducha-aiki/Ranger-Mish-ImageWoof-5/blob/master/mxresnet.py#L121)  

Activation function - Mish, long disscussion [on fastai forum](https://forums.fast.ai/t/meet-mish-new-activation-function-possible-successor-to-relu)  
Fit with Ranger optimizer and flat with annealing - [long tread on fastai forum](https://forums.fast.ai/t/how-we-beat-the-5-epoch-imagewoof-leaderboard-score-some-new-techniques-to-consider)  

Model was created with [model-constructor](https://github.com/ayasyrev/model_constructor)  
Explanation how model was created here: [notebook: ResnetTrick_create_model_fit.ipynb](https://github.com/ayasyrev/imagenette_experiments/blob/master/Imagenette_Nbs_1/ResnetTrick_create_model_fit.ipynb)

That experiments was with fastai version v1, explanation on page "First version".  
All notebooks can be run on google colab.  
