# realistic-ssl-evaluation

This repository contains the code for
[Realistic Evaluation of Deep Semi-Supervised Learning Algorithms](https://arxiv.org/abs/1804.09170), by Avital Oliver\*, Augustus Odena\*, Colin Raffel\*, Ekin D. Cubuk, and Ian J. Goodfellow, arXiv preprint arXiv:1804.09170.

If you use the code in this repository for a published research project, please cite this paper.

The code is designed to run on Python 3 using the dependencies listed in `requirements.txt`.
You can install the dependencies by running `pip3 install -r requirements.txt`.

The latest version of this repository can be found
[here](https://github.com/brain-research/realistic-ssl-evaluation).

# Prepare datasets

For SVHN and CIFAR-10, we provide scripts to automatically download and preprocess the data.
We also provide a script to create "label maps", which specify which entries of the dataset should be treated as labeled and unlabeled. Both of these scripts use an explicitly chosen random seed, so the same dataset order and label maps will be created each time. The random seeds can be overridden, for example to test robustness to different labeled splits.
Run those scripts as follows:

```sh
python3 build_tfrecords.py --dataset_name=cifar10
python3 build_label_map.py --dataset_name=cifar10
python3 build_tfrecords.py --dataset_name=svhn
python3 build_label_map.py --dataset_name=svhn
```

For ImageNet 32x32 (only used in the fine-tuning experiment), you'll first need to download the 32x32 version of the ImageNet dataset by following the instructions [here](https://patrykchrabaszcz.github.io/Imagenet32/).
Unzip the resulting files and put them in a directory called 'data/imagenet_32'.
You'll then need to convert those files (which are pickle files) into .npy files.
You can do this by executing:

```sh
mkdir data/imagenet_32
unzip Imagenet32_train.zip -d data/imagenet_32
unzip Imagenet32_val.zip -d data/imagenet_32
python3 convert_imagenet.py
```

Then you can build the TFRecord files like so:

```sh
python3 build_tfrecords.py --dataset_name=imagenet_32
```

ImageNet32x32 is the only dataset which must be downloaded manually, due to licensing issues.

# Running experiments

All of the experiments in our paper are accompanied by a .yml file in `runs/`.These .yml files are intended to be used with [https://github.com/tmux-python/tmuxp](tmuxp), which is a session manager for tmux.
They essentially provide a simple way to create a tmux session with all of the relevant tasks running (model training and evaluation).
The .yml files are named according to their corresponding figure/table/section in the paper.
For example, if you want to run an experiment evaluating VAT with 500 labels as shown in Figure 3, you could run

```sh
tmuxp load runs/figure-3-svhn-500-vat.yml
```

Of course, you can also run the code without using tmuxp.
Each .yml file specifies the commands needed for running each experiment.
For example, the file listed above `runs/figure-3-svhn-500-vat.yml` runs

```sh
CUDA_VISIBLE_DEVICES=0 python3 train_model.py --verbosity=0 --primary_dataset_name='svhn' --secondary_dataset_name='svhn' --root_dir=/mnt/experiment-logs/figure-3-svhn-500-vat --n_labeled=500 --consistency_model=vat --hparam_string=""  2>&1 | tee /mnt/experiment-logs/figure-3-svhn-500-vat_train.log
CUDA_VISIBLE_DEVICES=1 python3 evaluate_model.py --split=test --verbosity=0 --primary_dataset_name='svhn' --root_dir=/mnt/experiment-logs/figure-3-svhn-500-vat --consistency_model=vat --hparam_string=""  2>&1 | tee /mnt/experiment-logs/figure-3-svhn-500-vat_eval_test.log
CUDA_VISIBLE_DEVICES=2 python3 evaluate_model.py --split=valid --verbosity=0 --primary_dataset_name='svhn' --root_dir=/mnt/experiment-logs/figure-3-svhn-500-vat --consistency_model=vat --hparam_string=""  2>&1 | tee /mnt/experiment-logs/figure-3-svhn-500-vat_eval_valid.log
CUDA_VISIBLE_DEVICES=3 python3 evaluate_model.py --split=train --verbosity=0 --primary_dataset_name='svhn' --root_dir=/mnt/experiment-logs/figure-3-svhn-500-vat --consistency_model=vat --hparam_string=""  2>&1 | tee /mnt/experiment-logs/figure-3-svhn-500-vat_eval_train.log
```

Note that these commands are formulated to write out results to `/mnt/experiment-logs`.
You will either need to create this directory or modify them to write to a different directory.
Further, the .yml files are written to assume that this source tree lives in `/root/realistic-ssl-evaluation`.

## A note on reproducibility

While the focus of our paper is reproducibility, ultimately exact comparison to the results in our paper will be conflated by subtle differences such as the version of TensorFlow used, random seeds, etc.
In other words, simply copying the numbers stated in our paper may not provide a means for reliable comparison.
As a result, if you'd like to use our implementation of baseline methods as a point of comparison for e.g. a new semi-supervised learning technique, we'd recommend re-running our experiments from scratch in the same environment as your new technique.

# Simulating small validation sets

The following command runs evaluation on a set of checkpoints, with multiple resamples of small
validation sets (as in figure 5 in the paper):

```sh
python3 evaluate_checkpoints.py --primary_dataset_name='cifar10' --checkpoints='/mnt/experiment-logs/section-4-3-cifar-fine-tuning/default/model.ckpt-1000000,/mnt/.../model.ckpt-...,...'
```

Results are printed to stdout for each evaluation run, and at the end a string representation of the entire list
of validation accuracies for each resampled validation set and each checkpoint is printed:

```
{'/mnt/experiment-logs/table-1-svhn-1000-pi-model-run-5/default/model.ckpt-500001': [0.86, 0.93, 0.92, 0.91, 0.9, 0.94, 0.91, 0.88, 0.88, 0.89]}
```

# Disclaimer

This is not an official Google product.
