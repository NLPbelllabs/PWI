### Introduction 

This repository contains code for the paper "<b>What Features of Tags Strengthen Vision-Language Alignment? Broader Understanding from Semantic-, Phonological- and Bilingualism-Relatedness</b>"

We adopt [VisualBERT](https://github.com/uclanlp/visualbert) to investigate tags' functionality from the perspective of semantic-, phonological- and bilingual-relatedness, where the original tag of the input triple is replaced by a word with various features to probe its influence on VQA task performance.

### Pre-training

We provide jupyter notebook 'VLP_Pretaining.ipynb' for pre-training. Monolingual and bilingual pre-training, fine-tuning and VQA modeling all use different configurations in 'config'. Please change the 'mode' setting in 'src/param.py' for the different operations. 

### Fine-tuning

'VQA_Finetune.ipynb' is provided for monolingual or bilingual fine-tuning

### VQA modeling

Jupyter notebooks for the three experiments carried out in the paper are provided:

'PWI_VLP_Semantics.ipynb' - Semantic Relatedness

'PWI_VLP_Phonology.ipynb' - Phonological Relatedness

'PWI_VLP_Bilingual.ipynb' - Bilingual Relatedness

### Plot results

'Plot_PWI_Results.ipynb'
