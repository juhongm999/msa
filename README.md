## Memory-Based Sequential Attention

#### Directories

- `src` : contains all the python scripts for the project
- `ckpt` : where model checkpoints are saved to and loaded from (not included)
- `configs` : yaml files with architecture and hyperparameter details
- `logs` : output files recording training progress
- `notebooks` : miscellaneous visualizations 
- `plots` : output plots generated for a given model

Note that the hyperparameter trials are often saved under `*/hyper/*` and the best found are under `*/best/*`, but may cross-reference each other so that duplicate files are found saved.

#### Training

A configuration file, `__config__.yaml` should be written such that it can be used for training. The dataset details, hyperparameters, and architecture details are saved in this file. However, the actual optimizer and learning rate scheduler are defined in `trainer.py`. **Note:** training the best models found in our paper should reference the configs under `../configs/best/__dataset__/ours_*.yaml`. We can call the main script as,

```bash
nohup python main.py -c ../configs/__config__.yaml > ../logs/ours_029_2.out 2>&1 &

# usage parameters
usage: main.py [-h] -c path [--test]

experimental configuration

optional arguments:
  -h, --help            show this help message and exit
  -c path, --config path
                        the path to config file
  --test                test mode
```

#### Testing

Thereafter, we can test the model on the test data,

```bash
python main.py -c ../configs/best/cluttered/ours_008.yaml --test

# example output
Finished loading data: cluttered for testing with torch.Size([10000, 1, 60, 60]) samples.
[*] Loading model from ../ckpt/hyper/cluttered
[*] Loaded ours_008_model_best.pth.tar checkpoint @ epoch 366 with best valid acc of 93.790
array([0.2278, 0.5896, 0.7851, 0.8788, 0.9177, 0.9331])
----------------------------------------
Confusion Matrix:
    0    1    2    3    4    5    6    7    8    9  
  ---------------------------------------------------
0 | 960  2    0    1    2    5    8    3    4    3   
1 | 1    1061 9    1    11   3    6    17   3    7   
2 | 8    6    927  14   2    5    3    16   12   3   
3 | 2    2    13   976  2    16   1    5    12   7   
4 | 3    12   6    1    917  1    8    12   2    35  
5 | 1    1    4    27   1    818  19   5    8    8   
6 | 10   14   2    2    12   13   887  1    10   2   
7 | 3    31   12   9    9    2    0    956  3    17  
8 | 2    5    9    7    4    14   13   2    896  11  
9 | 3    3    3    2    24   9    2    20   15   933 

Metrics:
            0     1     2     3     4     5     6     7     8     9     mean
          -------------------------------------------------------------------
Precision | 0.967 0.933 0.941 0.938 0.932 0.923 0.937 0.922 0.928 0.909 0.933
Recall    | 0.972 0.948 0.931 0.942 0.920 0.917 0.931 0.917 0.930 0.920 0.933
F1        | 0.969 0.941 0.936 0.940 0.926 0.920 0.934 0.920 0.929 0.915 0.933

Overall Accuracy: 93.310 %
----------------------------------------
[*] test loss: 0.323 - test acc: 93.310 - test err: 6.690
```

#### Hyperparameter Tuning

The `hyper.py` file is used to generate and test hyperparameters. Values should be specified and updated within. Thereafter, we can generate the hyper files for a given dataset as,

```bash
python hyper.py --dataset cluttered --generate
```

To run all of the generated files, and to record the results to the `log` directory, we run the following:

```bash
nohup python hyper.py --dataset cluttered -p 6 >/dev/null 2>&1 &

# usage parameters
usage: hyper.py [-h] -d path [--log-dir path] [--config-dir path] [--generate] [-p [POOL_SIZE]] [--ram]

hyperparameter runner

optional arguments:
  -h, --help            show this help message and exit
  -d path, --dataset path
                        dataset to use
  --log-dir path        log directory
  --config-dir path     configs directory
  --generate            generate files
  -p [POOL_SIZE], --pool-size [POOL_SIZE]
                        number of processes
  --ram                 params for RAM
```

#### Plotting

Plots of the saved test samples, `plots/__dir__/test.npz`, (specified in `main.py`) can be visualized as follows:

```bash
python plot_glimpses.py -c ../configs/best/cluttered/ours_008.yaml --test -i 3 --pdf

# usage parameters
usage: plot_glimpses.py [-h] -c path [--animate] [--test] [--pdf] [-i CLASS_INDEX] [--epoch EPOCH]

plotter

optional arguments:
  -h, --help            show this help message and exit
  -c path, --config path
                        the path to config file
  --animate             animate
  --test                test mode
  --pdf                 save as pdf
  -i CLASS_INDEX, --class-index CLASS_INDEX
                        desired class index
  --epoch EPOCH         epoch of desired plot
```

#### Additional models

The fully-connected, convolutional, and vision transformer networks can be trained with the `default.py` file, which includes all our hyperparameters for each dataset. **Note:** these models rely on neural network code that is note included in this repository, but they are standard PyTorch implementations.

```bash
python default.py --dataset cluttered --model vit

usage: default.py [-h] -d path -m path

hyperparameter runner

optional arguments:
  -h, --help            show this help message and exit
  -d path, --dataset path
                        dataset to use
  -m path, --model path
                        model to train
```

#### Comments

All experiments are were conducted on an NVIDIA GeForce RTX 3090 (24GB), Intel i9-11900F @ 2.50GHz, and 128GB memory.