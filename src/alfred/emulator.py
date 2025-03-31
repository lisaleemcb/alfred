import os
import pickle
import scipy.interpolate
import numpy as np
import matplotlib.pyplot as plt
#import pandas as pd

import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
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

class Emulator:
    """A base class with for kSZ emulator."""
    def __init__(self, features=None, dataset=None, hyperparameters=None,
                 data_dir=f'/{home_dir}',
                 scale_data=True,
                 X_train=None,
                 X_test=None,
                 y_train=None,
                 y_test=None,
                 model=None,
                 uncertainties=None,
                 log_data=True,
                 method='Emulator_Base_Class',
                 verbose=True):
        
        self.dataset = dataset
        self.features = features
        self.hyperparameters = hyperparameters
        self.data_dir = data_dir
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test= y_test
        self.model = model
        if uncertainties is None:
            self.uncertainties = np.ones_like(self.dataset[0])
        elif uncertainties is not None:
            self.uncertainties = uncertainties
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

    def prep_data(self, splits=None, nells=30):
        """Describes the class."""
        
        if self.verbose:
            print('Prepping data for regresson')
            
        # self.params = np.asarray(self.dataset)
    
        # samples = np.zeros((len(self.dataset.index.tolist()), nells))
        # for i, sn in enumerate(self.dataset.index.tolist()):
        #     fn = f'{self.data_dir}/nells{nells}_v2/kSZ_LoReLi_simu{sn}.npz'
        #     spectra = np.load(fn)

        #     samples[i] = spectra['kSZ']

        if self.log_data:
            if self.verbose:
                print('logging input data...')
            self.dataset = np.log10(self.dataset)
                
        # self.ells = spectra['ells'] 
        # self.samples = samples
        # self.pn = np.where(np.isin(self.dataset.columns, self.features))[0]

        # if self.verbose:
        #     print(f'Prepped data')
        #     print(f'{samples.shape} samples')
        #     print(f'{self.params[:,self.pn].shape} features')

        if splits is None:
            X_train, X_test, y_train, y_test = train_test_split(self.features,
                                                                self.dataset,
                                                                test_size=0.2,
                                                                random_state=42)
        elif splits is not None:
            X_train, X_test, y_train, y_test = splits

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

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test= y_test

    def descale(self, arr, X_or_y='y'):
        if X_or_y == 'X':
            if self.scale_data:
                arr = self.scalerX.inverse_transform(arr)
            if self.log_data:
                arr = 10.0**arr
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

        if self.scale_data:
           y_test = self.scalerY.inverse_transform(self.y_test)

        if self.log_data:
            y_test = 10.0**y_test

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
    def __init__(self, dataset, features, hyperparameters, uncertainties=None,
                  scale_data=True, log_data=True, method='Neural Network', verbose=True):
        """First subclass with a unique implementation."""
        # Call the base class initializer to set up the common attributes
        super().__init__(features=features, dataset=dataset, hyperparameters=hyperparameters,
                            uncertainties=uncertainties, scale_data=scale_data, log_data=log_data,
                            method=method, verbose=verbose)

        self.neurons = hyperparameters['neurons']
        self.epochs = hyperparameters['epochs']

    def regress(self, kSZ=True, xe=False):
        if self.verbose:
            print(f"Now running regression with {self.neurons} neurons in the middle layer and {self.epochs} epochs...")

        
        if xe:
            def custom_loss(y_true, y_pred):
                loss = tf.reduce_mean(tf.losses.binary_crossentropy(y_true, y_pred))
                return loss + 0.01 * tf.reduce_sum(y_pred)  # Adding a small regularization term as an example


            layer_norm = tf.keras.layers.Normalization()
            layer_norm.adapt(self.X_train)

            nx = self.dataset.shape[1]
            # Define the model
            self.model = Sequential([
                layer_norm,
                Dense(nx, activation='relu', input_shape=(self.X_train.shape[1],)),
                Dense(nx, activation='relu'),
                Dense(nx, activation='relu'),
                #layers.Dense(nx, activation='relu'),
                Dense(nx, activation='relu'),
                Dense(nx, activation='linear')#'softmax' 'sigmoid'
            ])

            self.model.compile(optimizer='adam',loss='mse', metrics=['accuracy'])
            self.model.compile(optimizer='adam',loss='mse', metrics=[tf.keras.metrics.MeanSquaredError()])
            self.history = self.model.fit(self.X_train, self.y_train, epochs=600, batch_size=35, validation_data=(self.X_test, self.y_test))

        elif kSZ:
            def weighted_mse_loss(y_true, y_pred, uncertainty):
                # Compute squared error
                squared_error = tf.square(y_true - y_pred)
                
                # Weight by uncertainty (inverse of uncertainty)
                weights = tf.math.reciprocal(uncertainty)  # Higher uncertainty = lower weight
                
                # Apply the weights to the squared error
                weighted_error = squared_error * weights
                
                # Return the mean of the weighted error
                return tf.reduce_mean(weighted_error)

            uncertainty_tensor = tf.convert_to_tensor(self.uncertainties, dtype=tf.float32)
            
            # Train the model with validation data

            self.model = Sequential() #[Input(shape=input_shape), Dense(units=64, activation='relu')])
        # self.model.add(Dense(units=5, activation='relu')) # Input layer and first hidden layer (Dense layer)
            self.model.add(Dense(units=self.neurons, activation='leaky_relu')) # Second hidden layer
            self.model.add(Dense(units=self.neurons, activation='leaky_relu')) # Input layer and first hidden layer (Dense layer)
            self.model.add(Dense(units=self.neurons, activation='leaky_relu')) # Second hidden layer
            self.model.add(Dense(units=self.dataset.shape[1], activation='linear'))      # Output layer (for regression, no activation function)
            
        #     # Compile the model
            self.model.compile(optimizer='adam', loss=lambda y_true, y_pred: weighted_mse_loss(y_true, y_pred, uncertainty_tensor))
            
            # Train the model
            self.history = self.model.fit(self.X_train,
                                            self.y_train,
                                            epochs=self.epochs,
                                            batch_size=32,
                                            validation_data=(self.X_test, self.y_test),
                                            verbose=0)
            
        return
                                    
        
    def metrics(self):
        # Evaluate the model
        loss = self.model.evaluate(self.X_test, self.y_test)
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
    
    def save(self, dir, path=f"{base_dir}/emulators"):
        print(base_dir)
        os.makedirs(f"{path}/{dir}")
        self.model.save(f"{path}/{dir}/model.keras")

        joblib.dump(self.scalerX, f"{path}/{dir}/scalerX.pkl")
        joblib.dump(self.scalerY, f"{path}/{dir}/scalerY.pkl")

        np.savez(f"{path}/{dir}/training_files", X_train=self.X_train, X_test=self.X_test, y_train=self.y_train, y_test=self.y_test)

        if self.verbose:
            print(f"Emulator files saved in {dir}")

    @classmethod
    def load(cls, dir, path=f"{base_dir}/emulators", load_data=False):
        scalerX = joblib.load(f"{path}/{dir}/scalerX.pkl")
        scalerY = joblib.load(f"{path}/{dir}/scalerY.pkl")
        model = tf.keras.models.load_model(f"{path}/{dir}/model.keras")

        instance = cls(scalerX=scalerX, scalerY=scalerY, model=model)

        if load_data:
            X_train = np.load(f"{path/{X_train}}")
            X_test  = np.load(f"{path/{X_test}}")
            y_train  = np.load(f"{path/{y_train}}")
            y_test  = np.load(f"{path/{y_test}}")

            instance.X_train = X_train
            instance.X_test = X_test
            instance.y_train = y_train
            instance.y_test = y_test

        return instance

class RandomForest(Emulator):
    def __init__(self, dataset, hyperparameters, features, method='Random Forest', verbose=True):
        """First subclass with a unique implementation."""
        # Call the base class initializer to set up the common attributes
        super().__init__(dataset=dataset, hyperparameters=hyperparameters, 
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
