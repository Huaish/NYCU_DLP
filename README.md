# NYCU DLP Lab

## Environment Setup

### Install miniconda

Install miniconda

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
```

Initialize conda

```bash
~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh
```

```bash
source ~/.bashrc
source ~/.zshrc
```


### Create conda environment

```bash
conda create -n dlp python=3.8
conda activate dlp
```

Add conda-forge channel

```bash
conda config --add channels conda-forge
```

```bash
conda config --set auto_activate_base false
```
> Note: When using vscode, you may need to turn off auto_activate_base to prevent multiple python environments from being activated.


## Visualize(Tensorboard)

For visualizing the training process, we use tensorboard.

```bash
tensorboard --logdir logs --port 6007 --host=127.0.0.1 --port 6007 --reload_multifile True
```

Then, open your browser and go to [localhost:6007](localhost:6007) to see the training process.