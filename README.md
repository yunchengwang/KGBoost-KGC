# KGBoost: A Classification-Based Knowledge Base Completion Method with Negative Sampling

## This is the Python implementation of [KGBoost](https://www.sciencedirect.com/science/article/pii/S0167865522000939)

### Requirements
You will need numpy and xgboost to run KGBoost.

     pip install -r requirements.txt

### Download Pre-trained Entity Embedding
To download the pre-trained embedding for entities 
required in KGBoost. Simply run:

     source download_pretrained.sh

Pre-trained TransE and RotatE are generated using
[this code](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding).

Pre-trained word embeddings for FB15k-237 can be obtained
[here](https://code.google.com/archive/p/word2vec/).

### Run KGBoost

The best configurations for KGBoost-R and KGBoost-T
in each dataset are specified in best_config.sh. 
Uncomment the models and the datasets you want to run.

     source best_config.sh


**Citation**

If you find the source codes useful, please consider citing our [paper](https://doi.org/10.1016/j.patrec.2022.04.001):

```
@article{wang2022kgboost,
  title={KGBoost: A Classification-Based Knowledge Base Completion Method with Negative Sampling},
  author={Wang, Yun-Cheng and Ge, Xiou and Wang, Bin and Kuo, C-C Jay},
  journal={Pattern Recognition Letters},
  year={2022},
  publisher={Elsevier}
}
```