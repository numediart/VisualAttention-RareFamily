# VisualAttention-RareFamily

We provide several codes to compute image saliency from the Rare family. The philosophy of those models is that a specific feature does not necessarily attract human attention, but what attracts it is a feature which is rare, thus surprising and difficult to learn. 
> ![Rariy & Saliency](images/rarity.jpg) 
> Initial image on the left and raw saliency map (probability for each pixel to attract human attention) on the right. No filtering or centred Gaussian applied here. 

## DeepRare2019 - (DR2019)
Rarity is computed on the deep features extracted by a VGG16 trained on ImageNET. No training is needed. This model is neither "feature-engineered saliency model" as features come from a DNN model, nor a DNN-based model as it needs no training on an eye-tracking dataset: the default ImageNET training of the provided VGG16 is used. It is thus a "deep-engineered" model.

#### Use DR2019
A full paper can be found here : [https://arxiv.org/abs/2005.12073](https://arxiv.org/abs/2005.12073) and here is the [Github Project page](https://github.com/numediart/VisualAttention-DeepRare2019) .

#### Cite DR2019
If you use DR2019, please cite :   
>  `@misc{matei2020visual,  
 title={Visual Attention: Deep Rare Features}, author={Mancas Matei and Kong Phutphalla and Gosselin Bernard}, year={2020}, eprint={2005.12073}, archivePrefix={arXiv}, primaryClass={cs.CV}}`  

#### Special strength of DR2019
* Fully generic model with no training needed. Just run it on your images!
* Works better than Rare2012 and any other feature-engieneered model and better than some DNN-based models on general images dataset (MIT, ...)
* Works better than any DNN-based model on one-odd-out datasets (like P3, O3, ...) and is always in top-3 withe feature-engineered models
* Let you check the contributions of different VGG16 layers to the final result
* Fast even when ran only on CPU
* Interesting also for compression applications as the saliency map is precise


## Rare 2012 - (R2012)

Rarity is computed on 1) color and 2) Gabor features. This model is a "feature-engineered saliency model".

#### Use R2012
A full paper can be found here : [Main Rare2012 paper](http://applications.umons.ac.be/docnum/c7b423fd-d183-486c-9cec-966066b9b364/342FA573-191D-4A8C-9D3B-5003A53289B0/rare2012.pdf) and here is the [Github Project page](https://github.com/numediart/VisualAttention-Rare2012) .

#### Cite R2012
If you use R2012, please cite :   
>  @article{riche2013rare2012,
  title={Rare2012: A multi-scale rarity-based saliency detection with its comparative statistical analysis},
  author={Riche, Nicolas and Mancas, Matei and Duvinage, Matthieu and Mibulumukini, Makiese and Gosselin, Bernard and Dutoit, Thierry},
  journal={Signal Processing: Image Communication},
  volume={28},
  number={6},
  pages={642--658},
  year={2013},
  publisher={Elsevier}
}  

#### Special strength of R2012
* Generic ans easy to use
* Better than R2007


## Rare 2007 - (R2007)
Rarity is computed only on color features. This model is a "feature-engineered saliency model".

#### Use R2007
A full paper can be found here : [Main Rare2007 paper](https://www.researchgate.net/profile/Matei_Mancas/publication/221559276_Relative_Influence_of_Bottom-Up_and_Top-Down_Attention/links/09e4150c1b7dc86ef2000000.pdf) and here is the [Github Project page](https://github.com/numediart/VisualAttention-Rare2007) .

#### Cite R2007
If you use R2007, please cite :   
> @inproceedings{mancas2008relative,
  title={Relative influence of bottom-up and top-down attention},
  author={Mancas, Matei},
  booktitle={International Workshop on Attention in Cognitive Systems},
  pages={212--226},
  year={2008},
  organization={Springer}
}

#### Special strength of R2007
* Generic ans easy to use
* Interesting for compression applications as it provides a precise saliency map
