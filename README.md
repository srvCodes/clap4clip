<div align="center">

# CLAP4CLIP: Continual Learning with Probabilistic Finetuning for Vision-Language Models [[Paper]](https://arxiv.org/pdf/2403.19137v2)

## ✨ ***Now accepted to NeurIPS 2024!*** ✨

<p align="center">
  <a href="#what-is-clap4clip">What is CLAP4CLIP?</a> •
  <a href="#get-going">Get going</a> •
  <a href="#what-is-in-this-repo">What is in this repo?</a> •
  <a href="#language-aware-knowledge">Language-aware knowledge</a> •
    <a href="#uncertainty-related-ablations">Uncertainty-related ablations</a> •
  <a href="#cite">Cite</a>
</p>
</div>

---

## What is CLAP4CLIP?

![alt text](https://github.com/srvCodes/clap4clip/blob/main/images/Slide13-1.png "Logo Title Text 1")

[CLAP4CLIP](https://arxiv.org/pdf/2403.19137v2) is a general probabilistic finetuning framework for the pre-trained CLIP model on downstream class-incremental learning tasks.

The framework is general because (as depicted below) it supports a diverse range of prompt styles including hand-crafted prompts like [Continual-CLIP](https://arxiv.org/abs/2210.03114), task-conditioned prompts like [CoOp](https://arxiv.org/abs/2109.01134), instance-conditioned prompts like [AttriCLIP](https://arxiv.org/abs/2305.11488), and multi-modal prompts like [MaPLe](https://arxiv.org/abs/2210.03117):

![alt text](https://github.com/srvCodes/clap4clip/blob/main/images/Screenshot%202024-05-24%20162355.png)


## Get going

Clone this github repository:
```
git clone https://github.com/srvCodes/clap4clip.git
cd clap4clip
mkdir ckpt/
```
- Download models: Download the pretrained ViT-B-16.pt and ViT-L-14.pt checkpoints to `ckpt/` directory. 

- Download datasets: We suggest following the [mammoth](https://github.com/aimagelab/mammoth) library to download all the datasets into the repo `datasets/`. Instructions for ImageNet-R can be found [here](https://github.com/muzairkhattak/multimodal-prompt-learning/blob/main/docs/DATASETS.md).


## What is in this repo?

This repo is designed with the aim of benchmarking various finetuning methods for class-incremental learning with the pre-trained CLIP model.

The instructions below depict how to run the models provided with the initial release on CIFAR100 (check the repo `scripts/` and edit ):

- CLAP4CLIP with hand-crafted prompts (our base CLAP model):
```
python3 main_incremental_submit.py --lasp --beta 15 --db_name cifar100 --use-vga --expandable-adapter --finetuning --finetune-epochs 2 --num-run 10 --compute-ece --compute-bwt --train_batch 32 --exemplar-selector random --root ../path_to_datasets/ --multi-gpu --gpus 0,1 --default-gpu 0 --model clclip_var --epochs 5 --forward-times 20 --arch ViT-B-16  --method er --variational
```
- Continual-CLIP (zero-shot):
```
python3 main_incremental_submit.py --db_name cifar100 --num-run 10 --compute-ece --compute-bwt --train_batch 32 --root ../path_to_datasets/ --multi-gpu --gpus 0,1 --default-gpu 0 --model clclip --arch ViT-B-16
```
- CLIP-Adapter:
```
python3 main_incremental_submit.py --db_name cifar100 --finetuning --finetune-epochs 2 --num-run 10 --compute-ece --compute-bwt --train_batch 32 --exemplar-selector random --root ../path_to_datasets/ --multi-gpu --gpus 0,1 --default-gpu 0 --model clip_adapter --epochs 5 --arch ViT-B-16 --method er
```
- CLAP with CoOp:
```
python3 main_incremental_submit.py --db_name cifar100 --num-run 10 --compute-ece --compute-bwt --train_batch 32 --root ../path_to_datasets/ --multi-gpu --gpus 0,1 --default-gpu 0 --model coop_variational --arch ViT-B-16
```
- CLAP with MaPLe:
```
python3 main_incremental_submit.py --db_name cifar100 --num-run 10 --compute-ece --compute-bwt --train_batch 32 --root ../path_to_datasets/ --multi-gpu --gpus 0,1 --default-gpu 0 --model maple_variational --arch ViT-B-16
```
- CoOp with Adapter (used in Fig. 3b in the paper):
```
python3 main_incremental_submit.py --db_name cifar100 --num-run 10 --compute-ece --compute-bwt --train_batch 32 --root ../path_to_datasets/ --multi-gpu --gpus 0,1 --default-gpu 0 --model coop_adapter --arch ViT-B-16
```

We plan to release the following models upon the acceptance of our paper:
- ~~CLAP4CLIP with support for CoOp/MaPLe~~ Now released!

## Language-aware knowledge

- Past-task distribution regularization (for reducing **forgetting** in general): Can be evoked by passing the argument `--lasp --beta` $\gamma$ where $\gamma$ is the loss weight used in Eq. (12) in our paper.
- Weight initialization (for reducing **stability gap**): Currently, controlled by commenting/uncommenting [this line](https://github.com/srvCodes/clap4clip/blob/main/classifier/continual_clip_variational.py#L99).

## Uncertainty-related ablations

In our paper, we show the out-of-the-box perks of uncertainty-aware modelling for the following two tasks:

### Post-hoc novel data detection (PhNDD)

- PhNDD is a post-hoc setting proposed in our paper for evaluating the novel data detection capabilities of a finetuning algorithm within the continual learning setting. To evoke this, simply pass the argument `--eval-ood-score` in the script.

### Exemplar selection
- For all but the zero-shot models, the repo implements the following exemplar selection criteria: Random, Herding, Entropy, Variance, Variance of entropy, and Energy scores. These can simply be evoked by passing the value `x` to the argument `--exemplar-selector`, where `x` can be {`random`, `icarl`, `entropy`, `variance`, `distance`, `var_entropy`, `energy`}.

## Cite

If you want to cite this framework feel free to use this preprint citation:

```bibtex
@inproceedings{jha2024clap4clip,
  title={{CLAP4CLIP}: Continual Learning with Probabilistic Finetuning for Vision-Language Models},
  author={Saurav Jha and Dong Gong and Lina Yao},
  booktitle={Thirty-eighth Conference on Neural Information Processing Systems},
  year={2024},
  url={https://arxiv.org/pdf/2403.19137}
}
```
