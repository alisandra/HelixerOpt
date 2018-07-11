# HelixerOpt
TensorFlow and DNA sequence analysis--learning by doing

Tested only with python3

Developed originally with TensorFlow 1.4, some has been tested with 1.9

Python package requirements are listed in requirements.txt (pip install -r requirements.txt)

## Tensor2Tensor like
For now, code is structured similarly, but not identically, to 
TensorFlow's Tensor2Tensor (https://github.com/tensorflow/tensor2tensor)

Program execution is coordinated through controller.py.

There's a touch more control this way.

## Genic Problems
The GeneCallingProblem class coordinates formatting and prepping of labelled 
data for training a neural network on gene structural prediction.

It needs as raw input matching sequence (.fasta) and annotation (.gff3) files;
and converts these into (~) one-hot vectors of the base pairs ('C', 'A', 'T', 'G')
and the annotation ('intergenic', 'intronic', 'untranslated region', 'coding sequence').

### Usage example

First you will need data, and you will need a
registered subclass (e.g. GeneCallingProblemTest) to point to
it. If you don't have your own .gff3 and .fa files to work with 
or if just want an exact example to follow, just run the following.

```
mkdir -p test/arabidopsis
cd test/arabidopsis
wget ftp://ftp.ensemblgenomes.org/pub/plants/release-39/gff3/arabidopsis_thaliana/Arabidopsis_thaliana.TAIR10.39.gff3.gz
wget ftp://ftp.ensemblgenomes.org/pub/plants/release-39/fasta/arabidopsis_thaliana/dna/Arabidopsis_thaliana.TAIR10.dna.toplevel.fa.gz
gunzip *
# names have to be an exact match (to allow for multiple genomes)
mv Arabidopsis_thaliana.TAIR10.39.gff3 Arabidopsis_thaliana.gff3
mv Arabidopsis_thaliana.TAIR10.dna.toplevel.fa Arabidopsis_thaliana.fa
cd ../..
```

#### data -> shards
This command takes several minutes on test data
```
# note that argument to --data_dir must exist
python controller.py --data_dir $HOME/path-for-data-output \
    --problem GeneCallingProblemTest --generate_data
```

#### training model
Takes hours on test data, but you can interrupt it for testing purposes
```
# note that parent directories in --train_dir argument must exist
python controller.py --data_dir $HOME/path-for-data-output \
    --train_dir $HOME/path-for-training/genic_test \
    --problem GeneCallingProblemTest --model=plain_cnn_model_fn_01 --train
```


you can also add `--eval_name=despcriptive_name_for_data` to control the naming for how the 
new evaluations get saved and are displayed in tensorboard

#### predicting
In the model function: 

predicting returns whatever is passed to the `predictions` argument of
`tf.estimator.EstimatorSpec` when the mode is set to predict
(i.e. after `if mode == tf.estimator.ModeKeys.PREDICT`)

It's a bit hacky, but to pass the input `dna_in` into the `score_n_spec_sigmoid` function 
(for context, etc...), just add it to `params` (`params['dna_in'] = dna_in`)

```
python controller.py --data_dir $HOME/path-for-data-output \
    --train_dir $HOME/path-for-training/genic_test \
    --problem GeneCallingProblemTest --model=plain_cnn_model_fn_01 --predict
```

This saves data as numpy arrays in pickle format.

#### preview
This is like predict, but can get any part of the model back out if necessary.
Can also work if the model is not yet trainable.

##### --available, AKA list what can be previewed

```
python controller.py --data_dir $HOME/path-for-data-output \
    --problem GeneCallingProblemTest --model=plain_cnn_model_fn_01 \
    --preview --available --pickle whats_available.pkl
```
Warning, this is a bit hackish, but certainly better than fishing it out entirely manually 

##### Actually get preview
```
python controller.py --data_dir $HOME/path-for-data-output --train_dir $HOME/path-for-training/genic_test \
   --problem GeneCallingProblemTest --model=plain_cnn_model_fn_01 \
   --preview --activations=conv1d_1/kernel:0,dense_1/bias:0 --weights=conv1d_2/kernel:0 \
   --pickle current_values.pkl
```

## Tissue Expression Problems

**this doesn't really work yet**

The TissueExpressionProblem currently needs line wise matches between binned coverage 
and DNA sequence. However, this needs to be made more flexible / will be replaced. 
