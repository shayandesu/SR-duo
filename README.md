**We unlock few-step generation in discrete diffusion language models via the underlying Guassian diffusion.**

<div align="center">
  <img src="https://github.com/s-sahoo/duo/blob/gh-pages/static/images/duo_schematic.png" width="60%">
</div>

In this repo, we release:
* **The DUO framework**
  1. Curriculum learning strategy to speed up training. [[Example]](#training)
  2. Discrete Consistency Distillation pipeline. [[Example]](#distillation)
  3. Greedy-tail sampler. [[Example]](#sampling)
* **Baseline implementations** [[Examples]](#baselines):
  1. Autoregressive Model.
  2. [MDLM](https://arxiv.org/abs/2406.07524): Sahoo et al., "Simple and Effective Masked Diffusion Language Model", NeurIPS 2024.
  3. [SEDD (absorb)](https://arxiv.org/abs/2310.16834): Lou et al., "Score Entropy Based Discrete Diffusion", ICML 2024.
  4. [D3PM (absorb)](https://arxiv.org/abs/2107.03006) Austin et al., "Structured Denoising Diffusion Models in Discrete State-Spaces", NeurIPS 2021.

<!-- <a name="code-organization"></a>
## Code Organization
1. ```main.py```: The main entry point for training / eval.
2. ```trainer_base.py```: Boiler plate trainer using pytorch lightning.
3. ```algo.py```: Algorithms such as DUO, MDLM, AR, SEDD, D3PM.
4. ```dataloader.py```: Dataloaders.
5. ```utils.py```: LR scheduler, logging, `fsspec` handling.
6. ```models/```: Denoising network architectures. Supports [DiT](https://arxiv.org/abs/2212.09748) and AR transformer.
7. ```configs/```: Config files for datasets/denoising networks/noise schedules/LR schedules.
8. ```scripts/```: Shell scripts for training/evaluation. -->


<a name="getting_started"></a>

# Getting Started

To get started, create a conda environment containing the required dependencies.

```bash
conda create -n duo python=3.12
conda activate duo
conda install nvidia/label/cuda-12.4.0::cuda-toolkit
pip install -r requirements.txt
pip install flash_attn==2.7.4.post1
pip install numexpr==2.8.4
pip install sympytorch==0.1.1
```

### :izakaya_lantern: Integral Cache [Important]
Curriculum Learning (`Sec. 4.1`) and Discrete Consistency Distillation (`Sec. 4.2`) require mapping Gaussian to discrete diffusion parameters via the Diffusion Transformation operator (`Sec. 3`), which involves computing an integral (dependent only on the tokenizer’s vocabulary size). To avoid slowing down training, we pre-compute and cache this integral. Cached operators for `bert-base-uncased` (LM1B) and `gpt2` (OWT) are in [`integral/`](integral). For other tokenizers, run: 
```
python utils.py --vocab_size=N # N is 10292 in snip case
```
where `N` is the vocabulary size of the tokenizer.

### Slurm scripts
Run `mkdir watch_folder` to create a directory to store slurm logs
and then run any script in [`scripts/`](scripts) as a slurm job:
```bash
sbatch scripts/ABC_XYZ.sh
```

# Training
<a name="training"></a>

* DUO
```
python main.py algo=duo
```

* AR
```
python main.py algo=ar
```

* SEDD
```
python main.py algo=sedd sampling.predictor=analytic
```

* MDLM
```
python main.py algo=mdlm
```

* D3PM
```
python main.py algo=d3pm eval.compute_generative_perplexity=False 
```



# Sampling & Eval
<a name="sampling"></a>
To compute test perplexity on the validtion set of OWT use [`scripts/eval_owt_duo.sh`](scripts/eval_owt_duo.sh) and for zero shot perplexities use [`scripts/zero_shot_duo.sh`](scripts/zero_shot_duo.sh).


To generate samples from a pre-trained model use one of the following command.
Set 
* `sampling.noise_removal=greedy` to use the "Greedy-tail sampler" (equivalent to nucleus sampling in AR models; see `Sec. 4.2` in the paper).
* `sampling.noise_removal=ancestral` for the standard ancestral sampling. This produces more diverse samples (higher entropy) but with worse generative perplexity.

We have realease the distilled model `s-sahoo/duo-distilled` and the un-distilled model `s-sahoo/duo` on [Huggingface](https://huggingface.co/collections/s-sahoo/duo-67f9ff8fde919224e5fbd875)🤗. To sample from a HF model, run the following command:
```bash
python main.py \
  mode=sample_eval \
  loader.batch_size=2 \
  loader.eval_batch_size=8 \
  data=openwebtext-split \
  algo=duo_base \
  algo.backbone=hf_dit \
  eval.checkpoint_path=s-sahoo/duo-distilled \
  sampling.steps=8 \
  sampling.num_sample_batches=1 \
  sampling.noise_removal=greedy \
  +wandb.offline=true 
```
We’ve also released checkpoints for the distilled `duo-distilled.ckpt` and the un-distilled model `duo.ckpt` trained on OWT in this [Google Drive folder](https://drive.google.com/drive/folders/1JpqFM8XRvifwIkjWPfMyuDvu41r1yk0t?usp=share_link). Download them and use the command in [`scripts/gen_ppl_owt_duo.sh`](scripts/gen_ppl_owt_duo.sh) while specifying the paths correctly.


# Baselines
<a name="baselines"></a>
We release the checkpoints for the baselines: SEDD, MDLM and AR trained on OpenWebText in this [Google Drive folder](https://drive.google.com/drive/folders/16LuuptK7Xfk-vzhQYZBZ0SA-B-BFluau?usp=sharing). Download the checkpoints: `ar.ckpt`, `mdlm.ckpt`, `sedd.ckpt` and specify the paths appropriately in the respective shell scripts:
* [`scripts/eval_owt_*.sh`](scripts/) for computing validation perplexity on OWT.
* [`scripts/gen_ppl_*.sh`](scripts/) for generating text samples and evaluating them.
* [`scripts/zero_shot_*.sh`](scripts/) for computing zero shot perplexities.
* [`scripts/train_*.sh`](scripts/) for training the models.

# Acknowledgements & Citation
This repository was built off of [MDLM's Github repository](https://github.com/kuleshov-group/mdlm). Cite our paper using:
```
@inproceedings{
sahoo2025the,
title={The Diffusion Duality},
author={Subham Sekhar Sahoo and Justin Deschenaux and Aaron Gokaslan and Guanghan Wang and Justin T Chiu and Volodymyr Kuleshov},
booktitle={Forty-second International Conference on Machine Learning},
year={2025},
url={https://openreview.net/forum?id=9P9Y8FOSOk}
}
```
