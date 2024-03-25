# CREAM
Official code for "Cross-modal Retrieval with Noisy Correspondence via Consistency Refining and Mining"

# Requirements
Please follow these commands:
```
conda create -n cream python=3.7
conda install scikit-learn
conda install nltk
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
pip install matplotlib
```
After creating the environment, you need to download ```punkt``` as follows:
```
python
>>> import nltk
>>> nltk.download()
> d punkt
```

Those pre-trained models can be downloaded from:
[https://drive.google.com/drive/folders/1-OFZ4k1x2F0d0tgBznffNQf1aoRRvDT6?usp=sharing]

After downloading those models, you need to place them as follows:

```
  |--model_ckpt
     |--cream_models
        |--cc152k
        |--coco_n2
        |--f30k_n2
        |-- ...
```

# Data
We follow NCR [https://github.com/XLearning-SCU/2021-NeurIPS-NCR/] to obtain image features and vocabularies.
After downloading the data, you need to place the folders as follows:
```
    |--data
       |--data
          |--cc152k_precomp
          |--coco_precomp
          |--f30k_precomp
       |--vocab
          |--cc152k_precomp_vocab.json
          |--coco_precomp_vocab.json
          |--f30k_precomp_vocab.json
```

# Training
``` python run.py --data_name=f30k_precomp --noise_ratio=0.2 --num_epochs=40 ```

You can change --noise_ratio=***0.2*** to ***0.4 | 0.6 | 0.8*** to conduct more experiments on Flickr30K.

``` python run.py --data_name=cc152k_precomp --num_epochs=40 ```

As CC152K is a real-world dataset, there is no need to set --noise_ratio.

``` python run.py --data_name=coco_precomp --noise_ratio=0.2 --num_epochs=20 ```

You can change --noise_ratio=***0.2*** to ***0.4 | 0.6 | 0.8*** to conduct more experiments on MS-COCO.

# Evaluating

``` python evaluation.py ```

This will evaluate all the models in the ```model_path="./model_ckpt/cream_models/"```. If you need to evaluate one model, just change ```model_path``` in ```evaluation.py```. 

# About Graph Matching

Those codes are placed in another repository [https://github.com/allenHearst/CREAM-Graph-Matching/].

# Citation
If you found our work useful, please cite this work as follows, thank you.
```
@article{ma2024cream,
	title={Cross-modal Retrieval with Noisy Correspondence via Consistency Refining and Mining},
	author={Ma, Xinran and Yang, Mouxing and Li, Yunfan and Hu, Peng and Lv, Jiancheng and Peng, Xi},
	journal={IEEE transactions on image processing},
	year={2024}
}
```
