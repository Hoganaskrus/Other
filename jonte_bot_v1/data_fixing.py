import yfinance as yf
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
import time
import numpy as np
from mxnet import nd, autograd, gluon
from mxnet.gluon import nn, rnn
import mxnet as mx
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import accuracy_score


def main():
    gen = False
    train_time = False
    model_ctx=mx.cpu()
    if gen:
        GS, friends, indicators = generate_data()
        GS = ARMIA_time(GS)
        GS.to_csv(r'C:\Users\Jonathan\PycharmProjects\jonte_bot_test\GS.csv')
    else:
        GS = pd.read_csv("GS.csv",index_col=0)
    print('Total dataset has {} samples, and {} features.'.format(GS.shape[0], GS.shape[1]))

    for s in GS.columns:
        GS[s].fillna(method='bfill', inplace = True)
    GS = (GS - GS.min()) / (GS.max() - GS.min())

    batch_size = 128
    n_batches = GS.shape[0] / batch_size
    VAE_data = GS.to_numpy()
    num_training_days = int(GS.shape[0]*0.65)
    selector = [x for x in range(VAE_data.shape[1]) if x != 1]
    train_iter = mx.io.NDArrayIter(data={'data': VAE_data[:num_training_days, selector]}, label={'label': VAE_data[:num_training_days, 1:2]}, batch_size=batch_size)
    test_iter = mx.io.NDArrayIter(data={'data': VAE_data[num_training_days:, selector]}, label={'label': VAE_data[num_training_days:, 1:2]}, batch_size=batch_size)
    n_hidden = 400  # neurons in each layer
    n_latent = 2
    n_layers = 3  # num of dense layers in encoder and decoder respectively
    n_output = GS.shape[1]-1
    net = VAE(n_hidden=n_hidden, n_latent=n_latent, n_layers=n_layers, n_output=n_output, batch_size=batch_size)
    net.collect_params().initialize(mx.init.Xavier(), ctx=mx.cpu())
    net.hybridize()
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': .01})
    print(net)
    n_epoch = 150
    print_period = n_epoch // 10
    start = time.time()

    if train_time:
        training_loss = []
        validation_loss = []
        context = mx.cpu(); model_ctx=mx.cpu()
        mx.random.seed(1719)
        for epoch in range(n_epoch):
            epoch_loss = 0
            epoch_val_loss = 0

            train_iter.reset()
            test_iter.reset()

            n_batch_train = 0
            for batch in train_iter:
                n_batch_train += 1
                data = batch.data[0].as_in_context(mx.cpu())
                with autograd.record():
                    loss = net(data)
                loss.backward()
                trainer.step(data.shape[0])
                epoch_loss += nd.mean(loss).asscalar()

            n_batch_val = 0
            for batch in test_iter:
                n_batch_val += 1
                data = batch.data[0].as_in_context(mx.cpu())
                loss = net(data)
                epoch_val_loss += nd.mean(loss).asscalar()

            epoch_loss /= n_batch_train
            epoch_val_loss /= n_batch_val

            training_loss.append(epoch_loss)
            validation_loss.append(epoch_val_loss)

            """if epoch % max(print_period, 1) == 0:
                print('Epoch {}, Training loss {:.2f}, Validation loss {:.2f}'.\
                      format(epoch, epoch_loss, epoch_val_loss))"""

        end = time.time()
        print('Training completed in {} seconds.'.format(int(end - start)))
        net.save_parameters('net.params')
    else:
        net.load_parameters('net.params')
    pred =[]
    for t in range(0,len(GS),batch_size):
        if len(VAE_data[t:t+batch_size,selector]) == batch_size:
            temppred = net(mx.nd.array(VAE_data[t:t+batch_size,selector]).as_in_context(model_ctx))
            pred.extend(temppred.reshape((batch_size,)).asnumpy())
        else:
            temp = np.vstack([VAE_data[t:t+batch_size,selector],VAE_data[-1,selector]])
            i = batch_size-len(VAE_data[t:t+batch_size,selector])
            for j in range(1,i):
                temp = np.vstack([temp, VAE_data[-1, selector]])
            temppred = net(mx.nd.array(temp).as_in_context(model_ctx))
            pred.extend(temppred.reshape((batch_size,)).asnumpy())
    pred = pred[:-i]
    GS['dunno'] = pred
    GS['dunno'] = (GS['dunno'] - GS['dunno'].min()) / (GS['dunno'].max() - GS['dunno'].min())

    # plt.figure(figsize=(14, 5), dpi=100)
    # plt.plot(GS['dunno'])
    # plt.plot(GS['Close'])
    # plt.legend()
    # plt.show()

    #--------------------------------------------------------------
    batch_size = 32
    sequence_length = 17
    n_batches = GS.shape[0] / batch_size
    GAN_data = GS.to_numpy()
    num_training_days = int(GS.shape[0] * 0.909) # 2380
    gan_num_features = GAN_data.shape[1]
    X = GAN_data[:num_training_days,:]
    Y = []
    for i in range(num_training_days):
        Y.append(GAN_data[i+17,1])

    # print(X[1,:])
    # print(X[18, 1])
    # print(Y[1])

    #(sequence,batch,input_size)

    X = np.reshape(X,(7*20,gan_num_features,sequence_length))
    Y = np.reshape(Y,(7*20,sequence_length))
    number_train = 100
    number_validation = 40

    train_iter = mx.io.NDArrayIter(data={'data':X[:,:,:number_train]},label={'label':Y[:,:number_train]})
    validation_iter = mx.io.NDArrayIter(data={'data':X[:,:,number_train:]},label={'label':Y[:,number_train:]})

    #-----------------------------------------------------------------
    #Generator
    netG = nn.Sequential()
    with netG.name_scope():

        # Add the 1D Convolutional layers
        netG.add(gluon.nn.Conv1D(32, kernel_size=5, strides=2))
        netG.add(nn.LeakyReLU(0.01))
        netG.add(gluon.nn.Conv1D(64, kernel_size=5, strides=2))
        netG.add(nn.LeakyReLU(0.01))
        netG.add(nn.BatchNorm())
        netG.add(gluon.nn.Conv1D(128, kernel_size=5, strides=2))
        netG.add(nn.LeakyReLU(0.01))
        netG.add(nn.BatchNorm())

        # Add the two Fully Connected layers
        netG.add(nn.Dense(220, use_bias=False), nn.BatchNorm(), nn.LeakyReLU(0.01))
        netG.add(nn.Dense(220, use_bias=False), nn.Activation(activation='relu'))
        netG.add(nn.Dense(1))

    #Discr
    netD = RNNModel(num_embed=gan_num_features, num_hidden=500, num_layers=1)

    loss = gluon.loss.L1Loss()
    #loss = gluon.loss.SoftmaxCrossEntropyLoss()
    netD.collect_params().initialize(mx.init.Xavier(), ctx=mx.cpu())
    netG.collect_params().initialize(mx.init.Xavier(), ctx=mx.cpu())

    trainerG = gluon.Trainer(netG.collect_params(), 'adam', {'learning_rate': 0.01})
    trainerD = gluon.Trainer(netD.collect_params(), 'adam', {'learning_rate': 0.05})

    real_label = mx.nd.ones((batch_size,), ctx=mx.cpu())
    fake_label = mx.nd.zeros((batch_size,), ctx=mx.cpu())
    metric = mx.metric.Accuracy()
    print(netD)
    print(netG)


def generate_data():
    GS = yf.download("GS", period="1d" ,auto_adjust = True, prepost = True,start="2010-01-01", end="2020-05-29")
    friends = yf.download("MS JPM WFC C BAC BCS HSBC", period="1d" ,group_by = 'ticker',auto_adjust = True, prepost = True,start="2010-01-01", end="2020-05-29")

    indicators = pd.read_csv("USD3MTD156N.csv")
    indicators['BAMLH0A0HYM2'] = pd.read_csv("BAMLH0A0HYM2.csv")['BAMLH0A0HYM2']
    indicators['T10YIE'] = pd.read_csv("T10YIE.csv")['T10YIE']
    indicators['UNRATE'] = pd.read_csv("UNRATE.csv")['UNRATE']
    indicators['VIXCLS'] = pd.read_csv("VIXCLS.csv")['VIXCLS']
    indicators['NIKKEI225'] = pd.read_csv("NIKKEI225.csv")['NIKKEI225']
    indicators['NASDAQCOM'] = pd.read_csv("NASDAQCOM.csv")['NASDAQCOM']

    # Create 7 and 21 days Moving Average
    GS['ma7'] = GS['Close'].rolling(window=7).mean()
    GS['ma21'] = GS['Close'].rolling(window=21).mean()

    # Create MACD
    GS['26ema'] = pd.DataFrame.ewm(GS['Close'], span=26).mean()
    GS['12ema'] = pd.DataFrame.ewm(GS['Close'], span=12).mean()
    GS['MACD'] = (GS['12ema'] - GS['26ema'])

    # Create Bollinger Bands
    GS['20sd'] = GS['Close'].rolling(20).std()
    GS['upper_band'] = GS['ma21'] + (GS['20sd'] * 2)
    GS['lower_band'] = GS['ma21'] - (GS['20sd'] * 2)

    # Create Exponential moving average
    GS['ema'] = GS['Close'].ewm(com=0.5).mean()

    # Create Momentum
    GS['momentum'] = GS['Close'].pct_change()

    #plt.figure(figsize=(14, 5), dpi=100)
    #plt.plot(GS['Close'])
    #plt.show()

    #Create fft
    close_fft = np.fft.fft(np.asarray(GS['Close'].tolist()))
    fft_df = pd.DataFrame({'fft': close_fft})
    fft_df['absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
    fft_df['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))

    fft_list = np.asarray(fft_df['fft'].tolist())
    fft_list_temp = np.copy(fft_list);
    fft_list_temp[3:-3] = 0
    GS['fft3'] = abs(np.fft.ifft(fft_list_temp))
    fft_list_temp = np.copy(fft_list);
    fft_list_temp[6:-6] = 0
    GS['fft6'] = abs(np.fft.ifft(fft_list_temp))
    fft_list_temp = np.copy(fft_list);
    fft_list_temp[9:-9] = 0
    GS['fft9'] = abs(np.fft.ifft(fft_list_temp))

    return GS, friends, indicators

def ARMIA_time(GS):
    series = GS['Close']
    model = ARIMA(series,order=(5,1,0))
    model_fit = model.fit(disp=0)
    print(model_fit.summary())

    X = series.values
    size = int(len(X) * 0.66)
    train, test = X[0:size], X[size:len(X)]
    history = [x for x in train[0::17]]
    predictions = list()
    for t in range(17,len(X),17):
        model = ARIMA(history, order=(5, 1, 0))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast(steps=17)
        yhat = output[0]
        predictions.extend(yhat)
        obs = X[t:t+17]
        history.extend(obs)
    # error = mean_squared_error(X[:-1], predictions)
    # print('Test MSE: %.3f' % error)
    # plt.figure(figsize=(12, 6), dpi=100)
    # plt.plot(X, label='Real')
    # plt.plot(predictions, color='red', label='Predicted')
    # plt.xlabel('Days')
    # plt.ylabel('USD')
    # plt.title('Figure 5: ARIMA model on GS stock')
    # plt.legend()
    # plt.show()
    print(len(predictions))
    predictions.append(predictions[-1])
    GS['ARIMA'] = predictions
    return GS

model_ctx=mx.cpu()
class VAE(gluon.HybridBlock):
    def __init__(self, n_hidden=400, n_latent=2, n_layers=1, n_output=784,batch_size=100, act_type='relu', **kwargs):
        self.soft_zero = 1e-10
        self.n_latent = n_latent
        self.batch_size = batch_size
        self.output = None
        self.mu = None
        super(VAE, self).__init__(**kwargs)

        with self.name_scope():
            self.encoder = nn.HybridSequential(prefix='encoder')

            for i in range(n_layers):
                self.encoder.add(nn.Dense(n_hidden, activation=act_type))
            self.encoder.add(nn.Dense(n_latent * 2, activation=None))

            self.decoder = nn.HybridSequential(prefix='decoder')
            for i in range(n_layers):
                self.decoder.add(nn.Dense(n_hidden, activation=act_type))
            self.decoder.add(nn.Dense(n_output, activation='sigmoid'))

    def hybrid_forward(self, F, x):
        h = self.encoder(x)

        mu_lv = F.split(h, axis=1, num_outputs=2)
        mu = mu_lv[0]
        lv = mu_lv[1]
        self.mu = mu

        eps = F.random_normal(loc=0, scale=1, shape=(self.batch_size, self.n_latent), ctx=model_ctx)
        z = mu + F.exp(0.5 * lv) * eps
        y = self.decoder(z)
        self.output = y

        KL = 0.5 * F.sum(1 + lv - mu * mu - F.exp(lv), axis=1)
        logloss = F.sum(x * F.log(y + self.soft_zero) + (1 - x) * F.log(1 - y + self.soft_zero), axis=1)
        loss = -logloss - KL
        return loss


class RNNModel(gluon.Block):
    def __init__(self, num_embed, num_hidden, num_layers, bidirectional=False,sequence_length=17, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.num_hidden = num_hidden
        with self.name_scope():
            self.rnn = rnn.LSTM(num_hidden, num_layers, input_size=num_embed, \
                                bidirectional=bidirectional, layout='TNC')

            self.decoder = nn.Dense(1, in_units=num_hidden)

    def forward(self, inputs, hidden):
        output, hidden = self.rnn(inputs, hidden)
        decoded = self.decoder(output.reshape((-1, self.num_hidden)))
        return decoded, hidden

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)


class TriangularSchedule():
    def __init__(self, min_lr, max_lr, cycle_length, inc_fraction=0.5):
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.cycle_length = cycle_length
        self.inc_fraction = inc_fraction

    def __call__(self, iteration):
        if iteration <= self.cycle_length * self.inc_fraction:
            unit_cycle = iteration * 1 / (self.cycle_length * self.inc_fraction)
        elif iteration <= self.cycle_length:
            unit_cycle = (self.cycle_length - iteration) * 1 / (self.cycle_length * (1 - self.inc_fraction))
        else:
            unit_cycle = 0
        adjusted_cycle = (unit_cycle * (self.max_lr - self.min_lr)) + self.min_lr
        return adjusted_cycle


class CyclicalSchedule():
    def __init__(self, schedule_class, cycle_length, cycle_length_decay=1, cycle_magnitude_decay=1, **kwargs):
        self.schedule_class = schedule_class
        self.length = cycle_length
        self.length_decay = cycle_length_decay
        self.magnitude_decay = cycle_magnitude_decay
        self.kwargs = kwargs

    def __call__(self, iteration):
        cycle_idx = 0
        cycle_length = self.length
        idx = self.length
        while idx <= iteration:
            cycle_length = math.ceil(cycle_length * self.length_decay)
            cycle_idx += 1
            idx += cycle_length
        cycle_offset = iteration - idx + cycle_length

        schedule = self.schedule_class(cycle_length=cycle_length, **self.kwargs)
        return schedule(cycle_offset) * self.magnitude_decay ** cycle_idx

class RandIter(mx.io.DataIter):
    def __init__(self, batch_size, ndim):
        self.batch_size = batch_size
        self.ndim = ndim
        self.provide_data = [('rand', (batch_size, ndim, 1, 1))]
        self.provide_label = []

    def iter_next(self):
        return True

    def getdata(self):
        #Returns random numbers from a gaussian (normal) distribution
        #with mean=0 and standard deviation = 1
        return [mx.random.normal(0, 1.0, shape=(self.batch_size, self.ndim, 1, 1))]

def detach(hidden):
    if isinstance(hidden, (tuple, list)):
        hidden = [i.detach() for i in hidden]
    else:
        hidden = hidden.detach()
    return hidden

if __name__ == '__main__':
    main()