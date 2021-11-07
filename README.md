# Inventive spelling
Public repo with code for a computational investigation of the teaching method "inventive spelling". This is a teaching method used in reading and writing skill acquisition in primary schools.
In germanophone countries, this method is known as the **Reichen-method** or simply as **Lesen durch Schreiben**.

## Computational model
We propose a computational model based on two recurrent neural networks that allows to imitiate certain aspects of the inventive spelling.
![Overview](https://github.com/jannisborn/inventive_spelling/blob/master/assets/overview.pdf "Overview of computational model for reading and writing acquisition based on inventive spelling.")
*Overview of computational model for reading and writing acquisition based on inventive spelling.*


### Installation

Set up a new `conda` env:
```
conda create --name invspell python=3.6
```

Install the requirements and the package in editable mode:
```sh
pip install -r requirements.txt
pip install -e .
```

### Model training

For example, to train the model on the `childlex` data, run:
```py
python scripts/run.py --data_dir ${PWD}/data --epochs 250 --print_step 1 --task childlex \
--batch_size 2500 --save_model 125 --test_size 0.05 --learn_type normal --reading True \
--optimization Adam --dropout 0.5 
```