# Partial Implementation for Lane2Seq

## *Note
This repo is an **unofficial** and **partial** implementation for the paper (just one of my homework), resulting limited accuracy and may include errors.


## Reference
1. Lane2Seq: Towards Unified Lane Detection via Sequence Generation, [Paper](https://arxiv.org/abs/2402.17172).
2. Simple Implementation of Pix2Seq, [Code](https://github.com/moein-shariatnia/Pix2Seq).
3. Tuning computer vision models with task rewards, [Paper](https://arxiv.org/abs/2302.08242), [Code](https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/reward_tune/detection_reward.py).
4. LLAMAS dataset, [Code](https://github.com/karstenBehrendt/unsupervised_llamas), [Paper](https://ieeexplore.ieee.org/document/9022318).


## Limits
1. Only for LLAMAS dataset
2. Only grayscale images from LLAMAS without data augmentation.
3. Only try for Anchor representation, no task label (\<seg\>, \<anchor\> and \<param\>), no start-point.


## Environment
```
python >= 3.9
pytorch
pytorch-lightning
timm

numpy
opencv-python
scipy

argparse
```

## Architecture
#### Network
![arch](assets/arch.png)
#### MLE Pretrain
![mle](assets/mle.png)
#### MFRL Tuning
![tune](assets/tune.png)


## Results (Bad, :()
#### MLE Pretrain
The model was trained with 20 epochs and took 2.7 hours, getting **91.84 F1 score** on validation.
![train_loss](assets/train_loss.png)
![train_f1](assets/train_f1.png)

#### MFRL Tuning
The model then was tuned with 60 epochs and took 20.1 hours, gettting **92.54 F1 score** on validation (0.7 upgrade).
![tune_loss](assets/tune_loss.png)
![tune_f1](assets/tune_f1.png)

#### Test
Testing using *next-token-prediction* gots bad results. **Something must be wrong with my LLAMAS data processing, but I got limited time to solve it.**
![test](assets/test.png)

| Data Scoure    | Stage             | Test F1(%) $\uparrow$ |
| -------------- | ----------------- | --------------------- |
| Original Paper | MLE Pretrain      | **97.05**             |
| Implementation | MLE Pretrain      | **52.62**             |
| Implemnetation | Pretrain + Tuning | **53.99（+1.37）**    |

#### Visualization (Anchors)
Some visualization results while testing. **From left to right:** image, GT, prediction.
![vis1](assets/vis1.png)
![vis2](assets/vis2.png)
![vis3](assets/vis3.png)
![vis4](assets/vis4.png)
![vis5](assets/vis5.png)


