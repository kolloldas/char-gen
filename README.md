# char-gen
Simple character RNN generation using [tf.contrib.learn.Experiment](https://www.tensorflow.org/api_docs/python/tf/contrib/learn/Experiment)

You can train the system to generate short character sequences like names and phrases. 

Get a list of baby names from Kaggle [here](https://www.kaggle.com/kaggle/us-baby-names/data)

# Requirements
* Tensorflow 1.3+
* Numpy 1.13+

# Usage
### Training

You can provide a CSV file as a dataset or a text file with single example per line. CSV files must have headers.
Train the system:
```
python gen.py --data_file=NationalNames.csv --column=Name --train_steps=10000
```

By default the system will use a single layer LSTM with hidden(cell) size 256 and embedding size 128. To change any parameters pass it as a comma separated values to the `hparams` flag:
```
python gen.py --data_file=NationalNames.csv --column=Name --train_steps=10000 \
              --hparams=hidden_size=512,cell_type=GRU,num_layers=2
```
Modifiable hyperparameters:

| Name | Variable | Default |
|------|----------|----------|
| Batch size | batch_size | 128 |
| Embedding size | embedding_size | 128 |
| Vocabulary size | vocab_size | 256 |
| Learning rate | learn_rate | 0.001 |
| Cell type | cell_type | LSTM |
| Number of layers | num_layers | 1 |

Before training the data is preprocessed and cached into a file. To rebuild the cache use the flag `--nocache`

### Generation

After training is completed simply run the python file to generate samples
```
python gen.py
```

To add a primer character sequence:
```
python gen.py --primer=Ad
```

Generate 10 samples:
```
python gen.py --num_samples=10
```

Limit to 6 characters:
```
python gen.py --maxlen=6
```


Please note that if you modify any of the default hyperparameters during training, you must provide the exact same values during generation using the `--hparams` flag
