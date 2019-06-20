import tensorflow as tf
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.utils import shuffle

# If you want run the convolutional GSN on Colab
#from google.colab import drive
#from google.colab import files
#drive.mount('/content/drive')


def load_dataset():
    # Parameters
    aminoacids = 700
    features = 57

    data = np.load('dataset_path')

    # Reshape data
    data = np.reshape(data, (data.shape[0], aminoacids, features))

    # Build dataset
    _onehot_X = data[:, :, 0:21]
    _onehot_X = swap_onehot(_onehot_X)
    Y = data[:, :, 22:30]
    terminals = data[:, :, 31:33]
    _solvent_accessibility = data[:, :, 33:35]
    s_profile = data[:, :, 35:57]

    X = np.concatenate([_onehot_X , terminals], axis=-1)
    X = np.concatenate([X, _solvent_accessibility ], axis=-1)
    X = np.concatenate([X, s_profile ], axis=-1)
    
    full_data = np.concatenate([X,Y], axis=2)
    full_data = shuffle(full_data)
    
    return full_data
  
def load_cb513():
    # Parameters
    aminoacids = 700
    features = 57

    data = np.load('dataset_path')

    # Reshape data
    data = np.reshape(data, (data.shape[0], aminoacids, features))

    # Build dataset
    _onehot_X = data[:, :, 0:21]
    _onehot_X = swap_onehot(_onehot_X)
    Y = data[:, :, 22:30]
    terminals = data[:, :, 31:33]
    _solvent_accessibility = data[:, :, 33:35]
    s_profile = data[:, :, 35:57]
    
    X = np.concatenate([_onehot_X , terminals], axis=-1)
    X = np.concatenate([X, _solvent_accessibility ], axis=-1)
    X = np.concatenate([X, s_profile ], axis=-1)
    return X, Y

def split_dataset(data):
    # Split and shuffle train set and divide train test validation
    train = shuffle(data[0:5600, :, :])
    test = data[5605:5877, :, :]
    validation = data[5877:6133, :, :]
    
    X_train = train[:, :, 0:47] #era 24
    Y_train = train[:, :, 47:55]
    X_test = test[:, :, 0:47]
    Y_test = test[:, :, 47:55]
    X_val = validation[:, :, 0:47]
    Y_val = validation[:, :, 47:55]
    
    print(X_train.shape, 'x train shape')
    print(Y_train.shape, 'y train shape')
    
    return X_train, Y_train, X_val, Y_val, X_test, Y_test  
  
def swap_onehot(data):
  swap_data = data[:,:,[0, 1, 3, 2, 5, 4, 7, 6, 8, 10, 9, 11, 13, 12, 15, 14, 16, 18, 17, 20, 19]]
  return swap_data
  

aminoacids = 700
features = 47 #22 + 2 + 2 # 22 features + 2 terminals + 2 solvent access
label_size = 8

input_size = features*aminoacids
output_size = []
n_filters = 128
w_filter = 5
w_pool = 5
channels_0 = features + label_size #32
channels_1 = n_filters
channels_2 = n_filters

in_w1 =  aminoacids * features
out_w1 = aminoacids * channels_1
in_w2 = channels_1 * 140
out_w2 = channels_2 * 140

walkbacks = 12
n_epoch = 300
batch_size = 16
n_batch_train = 350
n_batch_test = 17
n_batch_val = 16
after_1_conv = 700
learning_rate = 0.001
beta = 0.01

def binomial_distribution(prob_v):
    shape = tf.shape(prob_v)
    random_v = tf.random_uniform(shape=shape, minval=0, maxval=1, dtype='float32')
    binomial_v =  tf.where(tf.less(random_v, prob_v),
                     tf.ones(shape, dtype='float32'), tf.zeros(shape, dtype='float32'))

    return binomial_v
  
def binomial_draw_vec(p_vec, dtype='float32'):
  shape = tf.shape(p_vec)
  return tf.select(tf.less(tf.random_uniform(shape=shape, minval=0, maxval=1, dtype='float32'), p_vec), tf.ones(shape, dtype=dtype), tf.zeros(shape, dtype=dtype))

def input_corrupt(X, rate=0.3):
    a = binomial_draw(shape=tf.shape(X), p=1-rate)
    b = binomial_draw(shape=tf.shape(X), p=0.5)
    z = tf.zeros(tf.shape(X), dtype='float32')
    c = tf.select(tf.equal(a, z), b, z)
    return tf.add(tf.mul(X, a), c)   

def input_corrupt_test(X):
    shape = X.shape
    to_tensor = np.zeros([batch_size, shape[1], shape[2]])
    for i in range(shape[0]):
        for j in range(shape[1]):
            r_ind = np.random.randint(0,label_size,1)
            to_tensor[i, j, r_ind[0]] = 1
    return to_tensor

def add_gaussian_noise(X):
    noise = tf.random_normal(tf.shape(X), stddev=2, dtype=tf.float32)
    return tf.add(X, noise)

def initialize_weights(w, c, n, n_in, n_out):
    interval = np.sqrt(6. / (n_in + n_out))
    weights = np.random.uniform(-interval, interval, size=(w,c,n))
    weights = np.float32(weights)
    return tf.Variable(weights)

def initialize_bias(c, offset = 0):
    bias = np.zeros([c]) - offset
    bias = np.float32(bias)
    return tf.Variable(bias)

def conv_1D_layer(data, w, b):
    H = tf.nn.conv1d(data, w, padding='SAME', stride=1)
    H = tf.add(H, b)
    H = tf.nn.tanh(H)
    return H

def conv_1D_layer_with_noise(data, w, b):
    H = tf.nn.conv1d(data, w, padding='SAME', stride=1)
    H = tf.add(H, b)
    H = add_gaussian_noise(tf.nn.tanh(add_gaussian_noise(H)))
    return H

def accuracy_Q8(real, pred):
    total = real.shape[0] * real.shape[1]
    correct = 0
    for i in range(real.shape[0]):  
        for j in range(real.shape[1]): 
            if np.sum(real[i, j, :]) == 0:
                total = total - 1
            else:
                if real[i, j, np.argmax(pred[i, j, :])] > 0:
                    correct = correct + 1
    return correct / total

def show_secondary(array):
    to_img = np.copy(array)
    for i in range(to_img.shape[0]):
        for j in range(to_img.shape[1]):
            for k in range(to_img.shape[2]):
                if to_img[i, j , k] == 1:
                    to_img[i, j, k] = 255
    #img = Image.fromarray(to_img)
    plt.figure(figsize=(28,10))
    plt.imshow(np.transpose(to_img[7,:,:]))
    plt.show()
    


''' Build the network '''
X_0 = tf.placeholder(tf.float32, shape=(batch_size, aminoacids, features), name='Pl_features')
Y_0 = tf.placeholder(tf.float32, shape=(batch_size, aminoacids, label_size), name="Pl_labels")
Y_labels = tf.placeholder(tf.float32, shape=(batch_size, aminoacids, label_size), name="Pl_labels_real")


H1_dec = tf.zeros([batch_size, after_1_conv, n_filters])

b0 = initialize_bias(channels_0)

w1 = initialize_weights(w_filter, channels_0, n_filters, in_w1, out_w1)
b1 = initialize_bias(channels_1)

w2 =  initialize_weights(w_filter, channels_2, n_filters, in_w2, out_w2)
b2 = initialize_bias(channels_2)

Y_samples  = []
predictions = []
Y_walkback = Y_0
logits = []

for i in range(walkbacks):
    Y_corrupt = input_corrupt(Y_walkback)

    input_data = tf.concat([X_0, Y_corrupt], 2)

    H1 = tf.add(tf.add(conv_1D_layer(input_data, w1, b=0), H1_dec), b1)

    AveragePool1D = tf.keras.layers.AveragePooling1D(pool_size = w_pool, padding = 'valid')
    H1_pool = AveragePool1D(H1)
    #H1_pool = tf.nn.dropout(H1_pool, keep_prob=0.7)  #dropout

    H2 = conv_1D_layer_with_noise(H1_pool, w2, b2)
    #H2 = tf.nn.dropout(H2, keep_prob = 0.7)  #dropout

    # Latent space

    # Deconv H2
    H2_dec = tf.nn.conv1d_transpose(H2, w2, output_shape=tf.shape(H1_pool), strides=(1))

    # Unpooling
    Upsampling1D = tf.keras.layers.UpSampling1D(size=5)
    H1_dec = Upsampling1D(H2_dec)
   
    lg = tf.add(tf.nn.conv1d_transpose(H1, w1, output_shape=tf.shape(input_data), strides=(1)), b0)
    lg_Y = lg[:, : , features:]
    logits.append(lg_Y)
    
    H_sample = tf.sigmoid(lg)

    Y_sample = H_sample[:, : , features:]

    Y_samples.append(Y_sample)

    Y_walkback = binomial_distribution(Y_sample)
    

cross_entropies = [tf.nn.sigmoid_cross_entropy_with_logits(labels=Y_labels, logits=lg) for lg in logits]
cross_entropy = tf.reduce_sum(tf.stack(cross_entropies))
reg = tf.nn.l2_loss(w1)+tf.nn.l2_loss(w2)
cross_entropy = (cross_entropy + beta * reg)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)


''' Network init '''

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

global_scores_train = []
global_scores_cb = []
global_loss_train = []
global_loss_cb = []

X_cb, Y_cb = load_cb513()
data = load_dataset()

for i in range(n_epoch):
  print('Epoch: ', i+1)
  train_cost = []
  cb_cost = []
  train_scores = []
  cb_scores = []
  
  data = shuffle(data)
  X_train, Y_train, X_val, Y_val, X_test, Y_test = split_dataset(data)
  
  X_train = np.concatenate([X_train, X_val, X_test], axis=0)
  Y_train = np.concatenate([Y_train, Y_val, Y_test], axis=0)
    
  # Training 
  for j in range(n_batch_train):
      batch_X_train = X_train[(j * batch_size):(j+1)*batch_size, :, :]
      batch_X_train = np.float32(batch_X_train)
      batch_Y_train = Y_train[(j * batch_size):(j+1)*batch_size, :, :]
      batch_Y_train = np.cast['float32'](batch_Y_train)
      train_dict = {X_0:batch_X_train, Y_0:batch_Y_train, Y_labels:batch_Y_train}
      ce, _, train_batch_pred = sess.run((cross_entropy, train_step, Y_sample), feed_dict=train_dict)
      train_cost.append(ce)
      train_scores.append(accuracy_Q8(batch_Y_train, train_batch_pred))

  train_loss = np.mean(train_cost)
  print('Train cost: ', train_loss/(batch_size*aminoacids*label_size))
  train_accuracy = np.mean(train_scores)
  print('Train Q8 accuracy:', train_accuracy)
  
  global_loss_train.append(train_loss/(batch_size*aminoacids*label_size))
  global_scores_train.append(train_accuracy)
  
  
  # Test on CB513 dataset
  for i in range(n_batch_val):
      batch_X_cb = X_cb[(i * batch_size):(i+1)*batch_size, :, :]
      batch_X_cb = np.float32(batch_X_cb)
      batch_Y_cb = np.cast['float32'](Y_cb[(i * batch_size):(i+1)*batch_size, :, :])
      corrupt_Y_cb = np.zeros([batch_size, aminoacids, label_size])
      corrupt_Y_cb = input_corrupt_test(corrupt_Y_cb)
      corrupt_Y_cb = np.float32(corrupt_Y_cb)
      cb_dict = {X_0:batch_X_cb, Y_0:batch_Y_cb, Y_labels:batch_Y_cb}
      ce, cb_batch_pred = sess.run((cross_entropy, Y_sample), feed_dict=cb_dict)
      cb_cost.append(ce)
      cb_scores.append(accuracy_Q8(batch_Y_cb, cb_batch_pred))
      if i == 0:
        show_secondary(batch_Y_cb)
        show_secondary(cb_batch_pred)
                           
  cb_loss = np.mean(cb_cost)
  print('cb513 cost: ', cb_loss/(batch_size*aminoacids*label_size))
  cb_accuracy = np.mean(cb_scores)
  print('cb513 Q8 accuracy:', cb_accuracy)
  print('------------------------------------------------------------------')

  global_loss_cb.append(cb_loss/(batch_size*aminoacids*label_size))
  global_scores_cb.append(cb_accuracy)
    
# Plot loss functions
fig = plt.figure()
plt.plot(range(n_epoch), global_loss_train, label='Training Loss')
plt.plot(range(n_epoch), global_loss_cb, label='CB513 Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
fig.savefig('loss_513.svg', format='svg', dpi=1200)

# Plot accuracy
fig = plt.figure()
plt.plot(range(n_epoch), global_scores_train, label='Training Q8 accuracy')
plt.plot(range(n_epoch), global_scores_cb, label='CB513 Q8 accuracy')
plt.xlabel('Epochs')
plt.ylabel('Q8 accuracy')
plt.legend()
plt.show()
fig.savefig('acc_513.svg', format='svg', dpi=1200)

# Print variance
print(np.std(global_scores_train), 'std sul train')
print(np.std(global_scores_cb), 'std sul test cb513')
