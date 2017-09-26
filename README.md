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
...
INFO:tensorflow:Prediction 1:  Ardara
INFO:tensorflow:Prediction 2:  Layton
INFO:tensorflow:Prediction 3:  Judgene
```

To add a primer character sequence:
```
python gen.py --primer=Ad
...
INFO:tensorflow:Prediction 1:  Adele
INFO:tensorflow:Prediction 2:  Adabell
INFO:tensorflow:Prediction 3:  Addie
```

Generate 10 samples:
```
python gen.py --num_samples=10
...
INFO:tensorflow:Prediction 1:  Dane
INFO:tensorflow:Prediction 2:  Merbie
INFO:tensorflow:Prediction 3:  Lorren
INFO:tensorflow:Prediction 4:  Rosette
INFO:tensorflow:Prediction 5:  Merlins
INFO:tensorflow:Prediction 6:  Curnebel
INFO:tensorflow:Prediction 7:  Spyrlette
INFO:tensorflow:Prediction 8:  Ronald
INFO:tensorflow:Prediction 9:  Selva
INFO:tensorflow:Prediction 10:  West
```

Limit to 4 characters:
```
python gen.py --maxlen=4
...
INFO:tensorflow:Prediction 1:  Giaz
INFO:tensorflow:Prediction 2:  Bee
INFO:tensorflow:Prediction 3:  Oran
```


Please note that if you modify any of the default hyperparameters during training, you must provide the exact same values during generation using the `--hparams` flag
