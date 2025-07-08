import os
import copy as cp
import pickle
import scipy.interpolate
import numpy as np
import matplotlib.pyplot as plt
#import pandas as pd

import json
import joblib
import keras
import tensorflow as tf

from tensorflow.keras import backend as K # for reproducibility
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, ELU, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import regularizers

from keras.saving import register_keras_serializable

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

from alfred.parameters import *


tf.config.set_visible_devices([], 'GPU')

def kemu(params, scalerX=None, scalerY=None, model=None, log_data=True):
    if params.ndim == 2:
        X = scalerX.transform(params)

    elif params.ndim == 1:
        X = scalerX.transform(params[None,:]) # the [None,:] is required by keras to maintain a rank 2 shape

    prediction = model.predict(X, verbose=0)
    Y = scalerY.inverse_transform(prediction) # this comes out as a rank 2

    if log_data:
        Y = 10**Y
        
    if params.ndim == 1:
        Y = Y.flatten()

    return Y

def mechkemu(params, emus, log_data=True):
    spectra = []

    for emu in emus:
        s = kemu(params, **emu, log_data=log_data)
        spectra.append(s)

    spectra = np.asarray(spectra)

    return spectra.mean(axis=0)

@register_keras_serializable(package="Custom")
class WeightedMSELoss(keras.losses.Loss):
    def __init__(self, ndata, **kwargs):
        super().__init__(**kwargs)
        self.ndata = ndata

    def call(self, y_true, y_pred):
        data_true = y_true[:, :self.ndata]   # Extract the true data values
        sigma_true = y_true[:, self.ndata:]  # Extract the uncertainty (sigma)

        mse = tf.reduce_mean(tf.square(y_pred - data_true))

        weights = 1.0 / (tf.square(sigma_true) + 1e-6)
        weighted_mse = tf.reduce_mean(weights * tf.square(y_pred - data_true))

        is_all_zeros = tf.reduce_all(tf.equal(sigma_true, 0.0))

        return tf.cond(is_all_zeros, lambda: mse, lambda: weighted_mse)
    
    def get_config(self):
        config = super().get_config()
        config.update({"ndata": self.ndata})
        return config

class Emulator:
    """A base class with for kSZ emulator."""
    def __init__(self, features=None,
                dataset=None,
                config=None,
                seed=None,
                splits=None,
                data_dir=f'/{home_dir}',
                scale_data=True,
                X_train=None,
                X_test=None,
                y_train=None,
                y_test=None,
                model=None,
                log_data=True,
                method='Emulator_Base_Class',
                verbose=True):

        if (dataset is None) == (splits is None):
            raise ValueError("Must provide either full dataset or split data (i.e. [X_train, X_test, y_train, y_test]).")
        
        if (dataset is None) != (features is None):
            raise ValueError("Must provide features corresponding to dataset")
        
        self.dataset = dataset
        self.features = features
        self.config = config
        self.seed = seed
        self.splits = splits
        self.data_dir = data_dir
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test= y_test
        self.model = model
        self.scale_data = scale_data
        self.log_data = log_data
        self.method = method
        self.verbose = verbose

        self.ells = None
        self.params = None
        self.samples = None
        self.model = None
        self.history = None
        self.scaler = None

    def prep_data(self, nells=30):
        """Describes the class."""
        
        if self.verbose:
            print('prepping data for regression...')
            
        # self.params = np.asarray(self.dataset)
    
        # samples = np.zeros((len(self.dataset.index.tolist()), nells))
        # for i, sn in enumerate(self.dataset.index.tolist()):
        #     fn = f'{self.data_dir}/nells{nells}_v2/kSZ_LoReLi_simu{sn}.npz'
        #     spectra = np.load(fn)

        #     samples[i] = spectra['kSZ']

        if self.splits is None:
            X_train, X_test, y_train, y_test = train_test_split(self.features,
                                                                self.dataset,
                                                                test_size=0.2,
                                                                random_state=self.seed,
                                                                shuffle=True)
        elif self.splits is not None:
            X_train, X_test, y_train, y_test = self.splits

        if self.log_data:
            if self.verbose:
                print('logging input data...')

            self.sigma_train = self.uncertainties / y_train
            self.sigma_test = self.uncertainties / y_test

            y_train = np.log10(y_train)
            y_test = np.log10(y_test)


        elif not self.log_data:
            self.sigma_train = self.uncertainties
            self.sigma_test = self.uncertainties
        
        if self.scale_data:
            if self.verbose:
                print('scaling features and data...')
  
        # Scale the data using StandardScaler
            self.scalerX = StandardScaler()
            X_train = self.scalerX.fit_transform(X_train)
            X_test = self.scalerX.transform(X_test)

            self.scalerY = StandardScaler()
            y_train = self.scalerY.fit_transform(y_train)
            y_test = self.scalerY.transform(y_test)

            self.sigma_train = cp.deepcopy(self.sigma_train / self.scalerY.scale_)
            self.sigma_test = cp.deepcopy(self.sigma_test / self.scalerY.scale_)


        self.ndata = y_train.shape[1]

        # print(f'train shape: {y_train.shape}')
        # print(f'test shape: {y_test.shape}')
        y_train = np.concatenate([y_train, self.sigma_train], axis=1)
        y_test = np.concatenate([y_test, self.sigma_test], axis=1)

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test= y_test

    def descale(self, arr, X_or_y='y'):
        if X_or_y == 'X':
            if self.scale_data:
                arr = self.scalerX.inverse_transform(arr)
            return arr
        
        if X_or_y == 'y':
            if self.scale_data:
                arr = self.scalerY.inverse_transform(arr)
            if self.log_data:
                arr = 10.0**arr
            return arr

       # return X_train, X_test, y_train, y_test

    def fancy_plot(self, xvariable):
        # plot Cls true and Cls predicted with errors

        y_test = self.descale(self.y_test[:,:self.ndata])
        y_pred = self.prediction(self.X_test)

        new_length = 9991
        new_xx = np.linspace(xvariable.min(), xvariable.max(), 11000)
        
        # pick random subset of 50 parameters among the test set 
        subset_indices = np.random.randint(low=0, high=y_test.shape[0], size=50)
        
        fig, axes = plt.subplots(3,1,figsize=(12,8),sharex=True,gridspec_kw={'height_ratios':(3,1,1),'hspace':0})
        axes[1].axhline(1,color='k',ls=':')
        for i, ind in enumerate(subset_indices):
            color = np.take(colors, i, mode='wrap')
            # plot true spectra for parameter set
            # rescale to true amplitudes (without alpha_i)
            ytrue = y_test[ind]
            axes[0].scatter(xvariable, ytrue, marker="+", color=color, s=30, zorder=2,alpha=.5)
            # plot predicted spectra for parameter set
            yrecons = y_pred[ind]
            # interpolated version to look nice
            new_yy = scipy.interpolate.interp1d(xvariable,y_pred[ind,:] , kind='quadratic', fill_value='extrapolate')(new_xx)
            #new_yy = new_yy*(exponents[0]*np.prod(np.abs(X_test[ind,:]/theta_ref)**exponents[1:]))
            axes[0].plot(new_xx,new_yy, "-", color=color, lw=1., zorder=1,alpha=.5)
            # ratio of pred to true
            axes[1].plot(xvariable,y_pred[ind,:]/y_test[ind,:],marker='o',color=color,lw=1,markersize=3,alpha=.5)
            # diff between pred and true
            axes[2].plot(xvariable,ytrue-yrecons,marker='o',color=color,lw=1,markersize=3,alpha=.5)
        
        # uncertainties on ratio
        ratio = y_pred/y_test
        a68, b68 = np.percentile(ratio,percentile1,axis=0), np.percentile(ratio,percentile2,axis=0)
        axes[1].fill_between(xvariable, b68, a68, color='k',alpha=0.2)
        axes[1].plot(xvariable,np.median(ratio,axis=0),color='k', linestyle='-', linewidth=2)
        # uncertainties on difference
        ratio2 = (y_test - y_pred)#*(exponents[0]*np.prod(np.abs(X_test/theta_ref)**exponents[1:],axis=1))[:,None]
        a682, b682 = np.percentile(ratio2,percentile1,axis=0), np.percentile(ratio2,percentile2,axis=0)
        axes[2].fill_between(xvariable, b682, a682, color='k',alpha=0.2)
        axes[2].plot(xvariable,np.median(ratio2,axis=0),color='k', linestyle='-', linewidth=2)
            
        axes[0].scatter([], [], marker="+", color='k', s=30, label='True values')
        axes[0].plot([], [], color='k', lw=1., label='Recovered')
        
        axes[0].legend(loc='best', fontsize=15)

        axes[0].set_ylim(0, 1.5)

        axes[1].set_ylim(0.5,1.25)
        axes[2].set_ylim(-0.1,0.1)
        axes[1].axhline(.88, color='red')
        axes[1].axhline(1.12, color='red')
       # axes[-1].set_xlabel(r"Angular multipole $\ell$", fontsize=15)
       # axes[0].set_ylabel(r"$C_\ell^{kSZ}$ [$\mu$K$^2$]", fontsize=15)
        axes[1].set_ylabel(r"Ratio", fontsize=15)
        axes[2].set_ylabel(r"Diff [$\mu$K$^2$]", fontsize=15)
        
        for i in range(len(axes)):
            axes[i].tick_params(labelsize=14)
        fig.tight_layout()

    def regress(self):
        """A method to be overridden by derived classes."""
        raise NotImplementedError("Subclasses must implement this method")

    def metrics(self):
        """A method to be overridden by derived classes."""
        raise NotImplementedError("Subclasses must implement this method")



class NeuralNetwork(Emulator):
    def __init__(self, config, seed=None, dataset=None, features=None, splits=None,
                  scale_data=True, log_data=True, method='Neural Network', verbose=True):
        """First subclass with a unique implementation."""
        # Call the base class initializer to set up the common attributes
        super().__init__(features=features, dataset=dataset, splits=splits, 
                        config=config, seed=seed,
                        scale_data=scale_data, log_data=log_data,
                        method=method, verbose=verbose)
        
        if (dataset is None) == (splits is None):
            raise ValueError("You must provide either a full dataset, or the splits of a dataset (in form [X_train, X_test, y_train, y_test])")

        self.config = config
        self.seed = seed
        self.features = features
        self.splits = splits
        self.neurons = self.config['neurons']
        self.epochs = self.config['epochs']
        self.nlayers = self.config['nlayers']
        self.uncertainties = self.config['uncertainties']
        self.batch_normalize = self.config['batch_normalize']
        self.add_dropout = self.config['add_dropout']
        self.early_stop = self.config['early_stop']
        self.reduce_lr = self.config['reduce_lr']
                
        self.ndata = None

        if self.uncertainties is None:
            if self.dataset is not None:
                self.uncertainties = np.zeros_like(self.dataset[0])
            elif self.splits is not None:
                self.uncertainties = np.zeros_like(splits[2][0])

        if self.seed is not None:
            if self.verbose:
                print(f"random seed is set to {self.seed}...")
            import random 

            os.environ['PYTHONHASHSEED'] = str(self.seed)
            random.seed(self.seed)
            np.random.seed(self.seed)
            tf.random.set_seed(self.seed)

            # torch.manual_seed(seed)
            # torch.cuda.manual_seed(seed)
            # torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.benchmark = True

            # Optional: Configure session for full reproducibility (slower!)
            # For TensorFlow 2.x and GPU use:
            os.environ['TF_DETERMINISTIC_OPS'] = '1'
            os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Optional: restrict GPU for simplicity


    def regress(self, kSZ=True):
        if self.verbose:
            print(f"running regression with config:")
            print(f"\t {self.config}")

        # Train the model with validation data
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

        loss_function = WeightedMSELoss(ndata=self.ndata)

        callbacks = []
        if self.early_stop:
            callbacks.append(early_stop)
        if self.reduce_lr:
            callbacks.append(reduce_lr)
        
        self.model = Sequential()
        self.model.add(Dense(units=self.neurons, kernel_regularizer=regularizers.l2(0.001)))
        self.model.add(LeakyReLU(negative_slope=0.01))
   

        for i in range(self.nlayers): # hidden layer
            reg_strength = min(1e-4 * 10**i, 1e-3)
            self.model.add(Dense(units=self.neurons, kernel_regularizer=regularizers.l2(reg_strength))) 
            if self.batch_normalize:
                self.model.add(BatchNormalization())
            self.model.add(LeakyReLU(negative_slope=0.01))
            if self.add_dropout:
                self.model.add(Dropout(0.2))
        
        self.model.add(Dense(units=self.ndata, activation='linear'))
       # optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)      # Output layer (for regression, no activation function)
        self.model.compile(optimizer='adam', loss=loss_function) # Compile the model
        
        verbose = 0
        if self.verbose:
            verbose = 1
        # Train the model

        # print(self.X_train.shape)
        # print(self.y_train.shape)
        # print(self.X_test.shape)
        # print(self.y_test.shape)
        self.history = self.model.fit(self.X_train,
                                        self.y_train,
                                        epochs=self.epochs,
                                        batch_size=64,
                                        callbacks=callbacks,
                                        validation_data=(self.X_test, self.y_test),
                                        verbose=verbose)
        
        return
    
    # @register_keras_serializable()
    # def weighted_mse_loss(y_true, y_pred, n_data=2):
    #     data_true = y_true[:, :self.n_data]   # Extract the true data values
    #     sigma_true = y_true[:, self.n_data:]  # Extract the uncertainty (sigma)

    #     error = tf.square(y_pred - data_true)
    #     weights = 1.0 / (tf.square(sigma_true) + 1e-6)
    #     weighted_error = weights * error

    #     return tf.reduce_mean(weighted_error)

    # def weighted_mse_loss(self, y_true, y_pred):
    #     sigma = self.sig
    #     error = tf.square(y_true - y_pred)

    #     return tf.reduce_mean(error / (tf.square(sigma) + 1e-6))

    def metrics(self):
        # Evaluate the model
        y_test = np.stack([self.y_test, self.uncertainties], axis=1)
        loss = self.model.evaluate(self.X_test, y_test)
        print(f"Test loss: {loss}")

        return

    def prediction(self, X):
    # Predict with the trained model, generally input X_test
        if self.scale_data:
            y = self.scalerY.inverse_transform(self.model.predict(X))
        else:
            y = self.model.predict(X)

        if self.log_data:
            y = 10**y

        return y
    
    def save(self, save_dir, base_path=f"{base_dir}/emulators"):
        path = f"{base_path}/{save_dir}"
        # Define directory
        os.makedirs(path, exist_ok=False)
        # Save the model
        self.model.save(os.path.join(path, "model.keras"))

        # Save the scalers
        joblib.dump(self.scalerX, os.path.join(path, "scalerX.pkl"))
        joblib.dump(self.scalerY, os.path.join(path, "scalerY.pkl"))

        # Save metadata (as a dict)
        np.savez(f"{path}/training_files", X_train=self.descale(self.X_train, X_or_y='X'),
                                                X_test=self.descale(self.X_test, X_or_y='X'),
                                                y_train=self.descale(self.y_train[:,:self.ndata]), 
                                                y_test=self.descale(self.y_test[:,:self.ndata]),
                                                seed=self.seed,
                                                config=self.config,
                                                uncertainties=self.uncertainties)

        if self.verbose:
            print(f"Emulator files saved in {path}.")

    @classmethod
    def load(cls, saved_dir, base_path=f"{base_dir}/emulators", load_data=False):
        path = f"{base_path}/{saved_dir}"
        from tensorflow.keras.models import load_model

        # Load components
        model = load_model(os.path.join(path, "model.keras"))
        scalerX = joblib.load(os.path.join(path, "scalerX.pkl"))
        scalerY = joblib.load(os.path.join(path, "scalerY.pkl"))
        metadata = np.load(os.path.join(path, 'training_files.npz'), allow_pickle=True)

        splits = None
        if load_data:
            print(metadata)
            X_train = metadata['X_train']
            X_test  = metadata['X_test']
            y_train  = metadata['y_train']
            y_test  = metadata['y_test']
            splits = [X_train, X_test, y_train, y_test]


        instance = cls(config=metadata['config'].item(), splits=splits)
        instance.scalerX = scalerX
        instance.scalerY = scalerY
        instance.model = model

        return instance

class RandomForest(Emulator):
    def __init__(self, dataset, config, features, method='Random Forest', verbose=True):
        """First subclass with a unique implementation."""
        # Call the base class initializer to set up the common attributes
        super().__init__(dataset=dataset, config=config, 
                         features=features, method=method, verbose=verbose)
        
    def regress(self, X_train, X_test, y_train, y_test):
        if self.verbose:
            print(f'Performing regression with {len(self.features)}:')
            print(f'\t {self.features}')
        self.regressor = RandomForestRegressor(n_estimators=100,         # Number of trees in the forest (default)
                                            max_depth=None,           # Fully grown trees (no max depth limit)
                                            min_samples_split=2,      # Minimum samples to split an internal node
                                            min_samples_leaf=1,       # Minimum samples required at a leaf node
                                            max_features=3,      # Number of features to consider when splitting (auto ≈ sqrt(num_features))
                                            random_state=42,          # Ensure reproducibility
                                            n_jobs=-1 )
        self.regressor.fit(X_train, y_train)

    def prediction(self, X):
    # Predict with the trained model, generally input X_test
        if self.scale_data:
            y = self.scalerY.inverse_transform(self.regressor.predict(X))
        else:
            y = self.regressor.predict(X)
        
        return y
    
            
    def metrics(self, X_test, y_test):
        # Evaluate the model
        loss = self.model.evaluate(X_test, y_test)
        print(f"Test loss: {loss}")

        return

def xe_emul(zvect, params, emul="keras_xe_emul", plot=False, H_He=1.08):

    '''
    zvect : vect of z values at which xe should be evaluated [z increasing]
    params: dict of params values 
            eg: params = {'fX':-2.71669877 , 
                          'rHS':0.2        , 
                           'tau': 3.51074603  , 
                           'Mmin':9.33       , 
                           'fesc': 0.275 }
    Emul: name of the directory containing the keras files
    allH: means H plus 1st reio of He

    RETURNS: xe values at zvect

    '''

    dir = "/Users/emcbride/Datasets/LoReLi/emulators/xe_emul"
    # data = np.load(f"{dir}/keras_xe_emul_pmean_pstd_zm_zs_xev.npy", allow_pickle=True)
    # model = keras.models.load_model(f'{dir}/keras_xe_emul')


    model = tf.keras.models.load_model(f"/Users/emcbride/alfred/model.keras")
    # model = tf.keras.models.load_model(f"{base_dir}/emulators/keras_xe_emul.keras")
    # data = np.load(f"{base_dir}/emulators/keras_xe_emul_pmean_pstd_zm_zs_xev.npy", allow_pickle=True)
    data = np.load(f"pmean_pstd_zm_zs_xev.npy", allow_pickle=True)

    parmeansstd = data.item()["parmeansstd"]
    zm = data.item()["zm"]    
    zs = data.item()["zs"]
    xe_interp = data.item()["xe_int"]


    X0_values = np.array([(params[key]-parmeansstd[key]["mean"])/parmeansstd[key]["std"] for key in params.keys()])


    Yval = model.predict(X0_values[None,:], verbose=0)

    zval = (Yval.T * zs + zm).flatten()[::-1]
    x_vals = np.hstack(([1e-1,0.98*zval[0]],zval.flatten()))
    x_vals = np.hstack((x_vals,[1.02*zval[-1]]))

                    
    y_vals = np.hstack(([1,1],xe_interp.flatten()[::-1]))
    y_vals = np.hstack((y_vals,[0.5*y_vals[-1]]))



    xe_fin = H_He * 10**(np.interp(np.log10(zvect), np.log10(x_vals), np.log10(y_vals), left=np.log10(1), right=-10))


    if plot:
        plt.figure()
        plt.plot(zvect, xe_fin)
        plt.xlabel("z")
        plt.ylabel(r"$x_e$")


    return xe_fin


    


        
# # Example usage
# if __name__ == "__main__":
#     # Create instances of the derived classes
#     obj1 = SubClass1("Object1")
#     obj2 = SubClass2("Object2")

#     # Call the describe method from the base class
#     obj1.describe()
#     obj2.describe()

#     # Call the overridden method1 from each derived class
#     obj1.method1()
#     obj2.method1()
