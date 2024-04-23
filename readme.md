# DNA: Differential Privacy Neural Augmentation for Contact Tracing

This repository contains the code for training Deep Set neural networks in the paper
'DNA: Differential Privacy Neural Augmentation for Contact Tracing' Rob Romijnders, Christos Louizos, Yuki M. Asano & Max Welling, to appear as Spotlight Talk in the ICLR workshop on Private ML.

The corresponding paper can be found on arxiv: [https://arxiv.org/abs/2404.13381](https://arxiv.org/abs/2404.13381).

The main training loop is in dpgnn/train.py. It's a standard training script for training in PyTorch. The main difference starts in line 230 where we manually calculate the singular values and vectors for each weight matrix, so as to ensure that the spectral norm is exactly 1/C.

# Contact
For questions or suggestions, please contact romijndersrob@gmail.com or r.romijnders@uva.nl

# Citation
If you use this code, please cite the following preprint:
```
@article{romijnders2021dna,
  title={DNA: Differential Privacy Neural Augmentation for Contact Tracing},
  author={Romijnders, Rob and Louizos, Christos and Asano, Yuki M and Welling, Max},
  journal={ICLR Workshop on Private ML},
  year={2024}
}
```
