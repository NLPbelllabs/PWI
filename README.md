




### Introduction 

This repository contains code for the paper "<b>Human-Like Distractor Response in Vision-Language Model</b>"

#### Abstract 
Previous studies exploring the human-like capabilities of machine-learning models have primarily focused on pure language models. 
Limited attention has been given to investigating whether models exhibit human-like behavior when performing tasks that require the integration of visual and language information.
In this study, we investigate the impact of tags of semantic, phonological, and bilingual features on the visual question answering task performance of an unsupervised model. 
Our findings reveal its similarities with the influence of distractors in the picture-naming task (known as the picture-word-interference paradigm) observed in human experiments: 
1) Semantically-related tags have a more negative effect on task performance compared to unrelated tags, indicating a more robust competition between visual and tag information which are semantically closer to each other when generating an answer. 
2) Even presenting a partial section (wordpiece) of the originally detected tag significantly improves task performance, with the portion that plays a lesser role in determining the overall meaning of the original tag leading to a more pronounced improvement. 
3) Tags in two languages that refer to the same meaning  exhibit a symmetrical-like effect on performance in balanced bilingual models.

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
