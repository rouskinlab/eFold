# eFold

This repo contains the pytorch code for our paper “*Diverse Database and Machine Learning Model to narrow the generalization gap in RNA structure prediction”* 

[[BioRXiv](https://www.biorxiv.org/content/10.1101/2024.01.24.577093v1.full)] [[Data](https://huggingface.co/rouskinlab)]

## Setup

### Download eFold

```bash
git clone https://github.com/rouskinlab/efold
```

### Create your environment

Using `virtualenv`

```bash
cd path/to/efold
python -m venv .venv
source .venv/bin/activate
pip install -m requirements.txt
```

Using conda

```bash
cd path/to/efold
conda create -n efold python=3.10
conda activate efold
pip install -m requirements.txt
```

### Install eFold

```bash
pip install path/to/efold
```

## File structure

```bash
bppm/ # bppm post processing step
efold/
    core/   # backend 
    models/ # where we define eFold and other models
scripts/
    efold_training.py # our training script
    [...]
best_efold/ # where we store our best model
    hyperparameters.json 
    weights.json
# python module boilerplate
LICENSE
requirements.txt
pyproject.toml
```

## Data

### List of the datasets we used

A breakdown of the data we used is summarized [here](https://github.com/rouskinlab/efold_data). All the data is stored on the [HuggingFace](https://huggingface.co/rouskinlab). 

### Get the data

You can download our datasets using [rouskinHF](https://github.com/rouskinlab/rouskinhf):

```bash
pip install rouskinhf
```

And in your code, write:

```python
>>> import rouskinhf
>>> data = rouskinhf.get_dataset('ribo500-blast') # look at the dataset names on huggingface
```

## Inference mode

### Using the command line

From a sequence:

```bash
efold --sequence AACCTGGUG -o seq.txt
cat seq.txt
#TODO
```

or a fasta file:

```bash
efold --fasta example.fasta -o seq.txt
cat seq.txt
#TODO
```

### Using python

```python
>>> from efold import inference
>>> inference(seq = 'AACUGUGCUA')
#TODO
```

## Reproducing our results

Run the training script:

```bash
cd path/to/efold
python scripts/efold_training.py
```

## Citation

**Plain text:**

Albéric A. de Lajarte, Yves J. Martin des Taillades, Colin Kalicki, Federico Fuchs Wightman, Justin Aruda, Dragui Salazar, Matthew F. Allan, Casper L’Esperance-Kerckhoff, Alex Kashi, Fabrice Jossinet, Silvi Rouskin. “Diverse Database and Machine Learning Model to narrow the generalization gap in RNA structure prediction”. bioRxiv 2024.01.24.577093; doi: https://doi.org/10.1101/2024.01.24.577093. 2024

**BibTex:**

```
@article {Lajarte_Martin_2024,
	title = {Diverse Database and Machine Learning Model to narrow the generalization gap in RNA structure prediction},
	author = {Alb{\'e}ric A. de Lajarte and Yves J. Martin des Taillades and Colin Kalicki and Federico Fuchs Wightman and Justin Aruda and Dragui Salazar and Matthew F. Allan and Casper L{\textquoteright}Esperance-Kerckhoff and Alex Kashi and Fabrice Jossinet and Silvi Rouskin},
	year = {2024},
	doi = {10.1101/2024.01.24.577093},
	URL = {https://www.biorxiv.org/content/early/2024/01/25/2024.01.24.577093},
	journal = {bioRxiv}
}

```
