from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import pandas as pd
import os
import six

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("data_file", None, "Name of data file (CSV)")
flags.DEFINE_string("column", None, "Column to pick from CSV")
flags.DEFINE_bool("nocache", False, "Do not load from cache file")
flags.DEFINE_integer("maxlen", 24, "Maximum character length")
flags.DEFINE_integer("train_steps", 0,
                     "The number of steps to run training for.")
flags.DEFINE_string("primer", "", "Primer characters during generation")
flags.DEFINE_integer("num_samples", 3, "Number of samples to generate")
flags.DEFINE_string("terminal", None, "Terminal character")

flags.DEFINE_string("hparams", "", "Comma separated list of hyperparameters")
flags.DEFINE_string("model_name", "char-gen", "Name of model")

cache_suffix = '-cache.npz'
_PAD = 0
_START = 1
RESERVED = 2

tf.logging.set_verbosity(tf.logging.INFO)

def decode_hparams(overrides=""):
    hp = tf.contrib.training.HParams(
        batch_size=128,
        embedding_size=128,
        vocab_size=256,
        hidden_size=256,
        learn_rate=0.001,
        cell_type='LSTM',
        num_layers=1
    )
    return hp.parse(overrides)

def make_cell(hparams):
    RNNCell = tf.nn.rnn_cell.GRUCell if hparams.cell_type == "GRU" else tf.nn.rnn_cell.LSTMCell
    rnn_cell = RNNCell(hparams.hidden_size)

    if hparams.num_layers > 1:
        rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_cell * hparams.num_layers)

    return rnn_cell

def lm_model_fn(features, labels, mode, params):
    hparams = params
    x = features['x']

    seq_size = x.get_shape()[1].value
    if seq_size is None:
        seq_size = tf.shape(x)[1]
    
    with tf.device("/cpu:0"):
        embedding = tf.Variable(tf.truncated_normal([hparams.vocab_size, hparams.embedding_size], 
                                                    stddev=1.0/np.sqrt(hparams.embedding_size)), 
                                name="embeddings")
        x_emb = tf.nn.embedding_lookup(embedding, x)

    cell = make_cell(hparams)

    
    with tf.variable_scope("lm"):
        outputs, _ = tf.nn.dynamic_rnn(cell, x_emb, dtype=tf.float32)
        outputs = tf.reshape(outputs, [-1, hparams.hidden_size])
        logits = tf.layers.dense(outputs, hparams.vocab_size, name="linear")
        
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        labels = tf.reshape(labels, [-1])
        loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)
        predictions = None
        optimizer = tf.contrib.opt.LazyAdamOptimizer(learning_rate=hparams.learn_rate)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    
    elif mode == tf.estimator.ModeKeys.PREDICT:
        loss = None
        train_op = None
        last_logit = tf.reshape(logits, [-1, seq_size, hparams.vocab_size])[:, -1, :]
        #print("Logits shape:", last_logit.get_shape())
        predictions = tf.nn.softmax(last_logit)

    return tf.contrib.learn.ModelFnOps(
            mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op
            )

def get_model_dir(model_name):
    model_dir = os.path.join(os.getcwd(), model_name)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    return model_dir

def create_estimator(hparams):
    return tf.contrib.learn.Estimator(
        model_fn=lm_model_fn, 
        model_dir=get_model_dir(FLAGS.model_name),
        params=hparams)

# Adapted from https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/data_generators/text_encoder.py
def encode(s):
    if six.PY2:
        if isinstance(s, unicode):
            s = s.encode("utf-8")
            return [ord(c) + RESERVED for c in s]
    # Python3: explicitly convert to UTF-8
    return [c + RESERVED for c in s.encode("utf-8")]

# Adapted from https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/data_generators/text_encoder.py
def decode(ids):
    decoded_ids = []
    int2byte = six.int2byte
    term = _PAD
    if FLAGS.terminal != None:
        term = encode(FLAGS.terminal)
    for id_ in ids:
        if 0 <= id_ < RESERVED:
            decoded_ids.append(b" ")
        else:
            decoded_ids.append(int2byte(id_ - RESERVED))
        
        if id_ == term:
            break

    if six.PY2:
      return "".join(decoded_ids)
    
    # Python3: join byte arrays and then decode string
    return b"".join(decoded_ids).decode("utf-8", "replace")

def convert_data(cache_file):
    assert FLAGS.data_file is not None, 'Data file must be provided'
    if FLAGS.column is not None:
        df = pd.read_csv(FLAGS.data_file, encoding='utf-8')
        data = df[FLAGS.column].values
    else:
        with open(FLAGS.data_file, 'r', encoding='utf-8') as fp:
            data = np.array(fp.read().splitlines(), dtype='O')

    x = []
    y = []

    for i in range(data.shape[0]):
        if i % 10000 == 0:
            print('Processed %d of %d items' % (i, len(data)))

        enc = encode(data[i][:FLAGS.maxlen])
        x.append([_START] + enc[:-1] + [_PAD] * (FLAGS.maxlen - len(enc)))
        y.append(enc + [_PAD] * (FLAGS.maxlen - len(enc)))

    x = np.array(x, dtype=np.int32)
    y = np.array(y, dtype=np.int32)
    print("Data shapes:", x.shape, y.shape)
    np.savez(cache_file, x=x, y=y)

def create_input_fns(hparams):
    cache_file = FLAGS.model_name + cache_suffix

    if not os.path.exists(cache_file) or FLAGS.nocache:
        convert_data(cache_file)

    npz = np.load(cache_file)

    train_input_fn = tf.estimator.inputs.numpy_input_fn (
        x={'x': npz['x']},
        y=npz['y'],
        batch_size=hparams.batch_size,
        num_epochs=None,
        shuffle=True
    )

    # Dummy
    eval_input_fn = tf.estimator.inputs.numpy_input_fn (
        x={'x': npz['x'][0]},
        y=npz['y'][0],
        batch_size=1,
        num_epochs=1,
        shuffle=False
    )

    return train_input_fn, eval_input_fn

def create_experiment(estimator, train_input_fn, eval_input_fn):
    return tf.contrib.learn.Experiment(
        estimator=estimator,
        train_input_fn=train_input_fn, 
        eval_input_fn=eval_input_fn,
        train_steps=FLAGS.train_steps
    )

def train():
    hparams = decode_hparams(FLAGS.hparams)
    tf.logging.info('HParams: %s', str(hparams))
    estimator = create_estimator(hparams)
    train_input_fn, eval_input_fn = create_input_fns(hparams)
    experiment = create_experiment(estimator, train_input_fn, eval_input_fn)
    experiment.train()

def generate():
    hparams = decode_hparams(FLAGS.hparams)
    tf.logging.info('HParams: %s', str(hparams))
    
    estimator = create_estimator(hparams)
    x = np.array([_START] + encode(FLAGS.primer), dtype=np.int32)
    x = np.tile(x, (FLAGS.num_samples, 1))

    vocab = np.arange(hparams.vocab_size)
    # Stop verbose
    tf.logging.info("Generating..")
    tf.logging.set_verbosity(tf.logging.ERROR)
    for i in range(FLAGS.maxlen):
        input_fn = tf.estimator.inputs.numpy_input_fn (
            x={'x': x},
            batch_size=FLAGS.num_samples,
            num_epochs=1,
            shuffle=False
        )
        y = estimator.predict(input_fn=input_fn, as_iterable=False)
        py = np.zeros([FLAGS.num_samples, 1], dtype=np.int32)
        for j in range(FLAGS.num_samples):
            py[j] = np.random.choice(vocab, p=y[j])
        
        x = np.concatenate([x, py], axis=1)
        

    tf.logging.set_verbosity(tf.logging.INFO)
    for j in range(FLAGS.num_samples):
        tf.logging.info("Prediction %d: %s", j+1, decode(x[j]))

def main(unused_argv):
    if FLAGS.train_steps > 0:
        train()
    else:
        generate()

tf.app.run()


