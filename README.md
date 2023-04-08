# Introduction
The FL training process comprises of two iterative phases, i.e., local training and global aggregation. Thus the learning performance is determined by both the effectiveness of the parameters from local training and smooth aggregation of them. However, these two requirements are not easy to satisfy in edge environment, i.e., edge users often have limited bandwidth and insufficient data, which can cause inefficient parameters aggregation, excessive training time and reduced model accuracy. FL inherently entails a large number of communication rounds and a large amount of labeled data for training, which are often unavailable for edge users. Such challenges are particularly salient under the combined effect of a long training process and unfavorable factors such as non-IID and unbalanced data, limited communication bandwidth, and unreliable and limited device availability.

We revisits the question of how FL mines the distributed data in iterative training rounds, and exploit the emerging foundation model (FM) to optimize the FL training. We investigate the behavior of the nascent model in a standard FL setting using popular off-the-shelf FMs, e.g., CLIP, and methods for FM adaptation. We propose PROMPTFL, a framework that replaces existing federated model training with prompt training, i.e., FL clients train prompts instead of a model, which can simultaneously exploit the insufficient local data and reduce the aggregation overhead. PROMPTFL ships an off-the-shelf public CLIP to users and apply continuous prompts (a.k.a. soft prompts) for FM adaptation, which requires very few data samples from edge users. The framework is technically very simple but effective.


## Citation

If this code is useful in your research, you are encouraged to cite our academic paper:
```
@article{guo2022promptfl,
  title={PromptFL: Let Federated Participants Cooperatively Learn Prompts Instead of Models--Federated Learning in Age of Foundation Model},
  author={Guo, Tao and Guo, Song and Wang, Junxiao and Xu, Wenchao},
  journal={arXiv preprint arXiv:2208.11625},
  year={2022}
}
```
