# need to install latest cleverhans (not via pip) for HopSkipJump
import numpy as np
import tensorflow as tf


def binary_rand_robust(model, ds, p, max_samples=100, noise_samples=10000, stddev=0.025, input_dim=[None, 107],
                num=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 250, 500, 1000, 2500, 5000, 10000],
                dataset='adult'):
  """Calculate robustness to random noise for Adv-x MI attack on binary-featureed datasets (+ UCI adult).

    :param model: model to approximate distances on (attack).
    :param ds: tf dataset should be either the training set or the test set.
    :param p: probability for Bernoulli flips for binary features.
    :param max_samples: maximum number of samples to take from the ds
    :param noise_samples: number of noised samples to take for each sample in the ds.
    :param stddev: the standard deviation to use for Gaussian noise (only for Adult, which has some continuous features)
    :param input_dim: dimension of inputs for the dataset.
    :param num: subnumber of samples to evaluate. max number is noise_samples
    :return: a list of lists. each sublist of the accuracy on up to $num noise_samples.
    """
  # switch to TF1 style
  sess = K.get_session()
  x = tf.placeholder(dtype=tf.float32, shape=input_dim)
  output = tf.argmax(model(x), axis=-1)
  next_element = ds.make_one_shot_iterator().get_next()

  num_samples = 0
  robust_accs = [[] for _ in num]
  all_correct = []
  while (True):
    try:
      xbatch, ybatch = sess.run(next_element)
      labels = np.argmax(ybatch, axis=-1)
      y_pred = sess.run(output, feed_dict={x: xbatch})
      correct = y_pred == labels
      all_correct.extend(correct)
      for i in range(len(xbatch)):
        if correct[i]:
          if dataset == 'adult':
            noise = np.random.binomial(1, p, [noise_samples, xbatch[i: i+1, 6:].shape[-1]])
            x_sampled = np.tile(np.copy(xbatch[i:i+1]), (noise_samples, 1))
            x_noisy = np.invert(xbatch[i: i+1, 6:].astype(np.bool), out=np.copy(x_sampled[:, 6:]), where=noise.astype(np.bool)).astype(np.int32)
            noise = stddev * np.random.randn(noise_samples, xbatch[i: i+1, :6].shape[-1])
            x_noisy = np.concatenate([x_sampled[:, :6] + noise, x_noisy], axis=1)
          else:
            noise = np.random.binomial(1, p, [noise_samples, xbatch[i: i + 1].shape[-1]])
            x_sampled = np.tile(np.copy(xbatch[i:i + 1]), (noise_samples, 1))
            x_noisy = np.invert(xbatch[i: i + 1].astype(np.bool), out=x_sampled,
                                where=noise.astype(np.bool)).astype(np.int32)
          preds = []

          bsize = 100
          num_batches = noise_samples // bsize
          for j in range(num_batches):
            preds.extend(sess.run(output, feed_dict={x: x_noisy[j * bsize:(j + 1) * bsize]}))

          for idx, n in enumerate(num):
            if n == 0:
              robust_accs[idx].append(1)
            else:
              robust_accs[idx].append(np.mean(preds[:n] == labels[i]))
        else:
          for idx in range(len(num)):
            robust_accs[idx].append(0)

      num_samples += len(xbatch)
      if num_samples >= max_samples:
        break

    except tf.errors.OutOfRangeError:
      break

  return robust_accs


def continuous_rand_robust(model, ds, max_samples=100, noise_samples=2500, stddev=0.025, input_dim=[None, 32, 32, 3],
                           num=[1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 75, 100, 150, 200, 350, 500, 750, 1000, 1500, 2000, 2500]):
  """Calculate robustness to random noise for Adv-x MI attack on continuous-featureed datasets (+ UCI adult).

  :param model: model to approximate distances on (attack).
  :param ds: tf dataset should be either the training set or the test set.
  :param max_samples: maximum number of samples to take from the ds
  :param noise_samples: number of noised samples to take for each sample in the ds.
  :param stddev: the standard deviation to use for Gaussian noise (only for Adult, which has some continuous features)
  :param input_dim: dimension of inputs for the dataset.
  :param num: subnumber of samples to evaluate. max number is noise_samples
  :return: a list of lists. each sublist of the accuracy on up to $num noise_samples.
  """
  # switch to TF1 style
  sess = K.get_session()
  x = tf.placeholder(dtype=tf.float32, shape=input_dim)
  output = tf.argmax(model(x), axis=-1)
  next_element = ds.make_one_shot_iterator().get_next()

  num_samples = 0
  robust_accs = [[] for _ in num]
  all_correct = []
  while (True):
    try:
      xbatch, ybatch = sess.run(next_element)
      labels = np.argmax(ybatch, axis=-1)
      y_pred = sess.run(output, feed_dict={x: xbatch})
      correct = y_pred == labels
      all_correct.extend(correct)

      x_adv_np = []
      for i in range(len(xbatch)):
        if correct[i]:
          noise = stddev * np.random.randn(noise_samples, input_dim[1:])
          x_noisy = np.clip(xbatch[i:i + 1] + noise, 0, 1)
          preds = []

          bsize = 50
          num_batches = noise_samples // bsize
          for j in range(num_batches):
            preds.extend(sess.run(output, feed_dict={x: x_noisy[j * bsize:(j + 1) * bsize]}))

          for idx, n in enumerate(num):
            if n == 0:
              robust_accs[idx].append(1)
            else:
              robust_accs[idx].append(np.mean(preds[:n] == labels[i]))
        else:
          for idx in range(len(num)):
            robust_accs[idx].append(0)

      num_samples += len(xbatch)
      # print("processed {} examples".format(num_samples))
      # print(correct)
      # print(robust_accs[-len(xbatch):])
      if num_samples >= max_samples:
        break


    # not sure how to iterate over a TF Dataset in TF1.
    # this is ugly but it works
    except tf.errors.OutOfRangeError:
      break

  return robust_accs
