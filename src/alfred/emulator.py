import os
import pickle
import scipy.interpolate
import numpy as np
import matplotlib.pyplot as plt
#import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

from alfred.parameters import *

home_dir = 'Users/emcbride'

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
    def __init__(self, features, dataset=None, hyperparameters=None,
                 data_dir=f'/{home_dir}',
                 scale_data=True,
                 log_data=True,
                 method='Emulator_Base_Class',
                 verbose=True):
        
        self.dataset = dataset
        self.features = features
        self.hyperparameters = hyperparameters
        self.data_dir = data_dir
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
            print('Prepping data for regresson')
            
        self.params = self.dataset.to_numpy()
    
        samples = np.zeros((len(self.dataset.index.tolist()), nells))
        for i, sn in enumerate(self.dataset.index.tolist()):
            fn = f'{self.data_dir}/nells{nells}_v2/kSZ_LoReLi_simu{sn}.npz'
            spectra = np.load(fn)

            samples[i] = spectra['kSZ']

        if self.log_data:
            if self.verbose:
                print('logging spectra data...')
            samples = np.log10(samples)
                
        self.ells = spectra['ells'] 
        self.samples = samples
        self.pn = np.where(np.isin(self.dataset.columns, self.features))[0]

        if self.verbose:
            print(f'Prepped data')
            print(f'{samples.shape} samples')
            print(f'{self.params[:,self.pn].shape} features')
            
        X_train, X_test, y_train, y_test = train_test_split(self.params[:,self.pn], samples, test_size=0.2, random_state=42)

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

        return X_train, X_test, y_train, y_test

    def fancy_plot(self, y_test, y_pred):
        # plot Cls true and Cls predicted with errors

        if self.scale_data:
            y_test = self.scalerY.inverse_transform(y_test)

        if self.log_data:
            y_test = 10.0**y_test

        new_length = 9991
        new_xx = np.linspace(1, 15000, 11000)
        
        # pick random subset of 50 parameters among the test set 
        subset_indices = np.random.randint(low=0, high=y_test.shape[0], size=50)
        
        fig, axes = plt.subplots(3,1,figsize=(12,8),sharex=True,gridspec_kw={'height_ratios':(3,1,1),'hspace':0})
        axes[1].axhline(1,color='k',ls=':')
        for i, ind in enumerate(subset_indices):
            color = np.take(colors, i, mode='wrap')
            # plot true spectra for parameter set
            # rescale to true amplitudes (without alpha_i)
            ytrue = y_test[ind]
            axes[0].scatter(self.ells,ytrue, marker="+", color=color, s=30, zorder=2,alpha=.5)
            # plot predicted spectra for parameter set
            yrecons = y_pred[ind]
            # interpolated version to look nice
            new_yy = scipy.interpolate.interp1d(self.ells,y_pred[ind,:] , kind='quadratic', fill_value='extrapolate')(new_xx)
            #new_yy = new_yy*(exponents[0]*np.prod(np.abs(X_test[ind,:]/theta_ref)**exponents[1:]))
            axes[0].plot(new_xx,new_yy, "-", color=color, lw=1., zorder=1,alpha=.5)
            # ratio of pred to true
            axes[1].plot(self.ells,y_pred[ind,:]/y_test[ind,:],marker='o',color=color,lw=1,markersize=3,alpha=.5)
            # diff between pred and true
            axes[2].plot(self.ells,ytrue-yrecons,marker='o',color=color,lw=1,markersize=3,alpha=.5)
        
        # uncertainties on ratio
        ratio = y_pred/y_test
        a68, b68 = np.percentile(ratio,percentile1,axis=0), np.percentile(ratio,percentile2,axis=0)
        axes[1].fill_between(self.ells, b68, a68, color='k',alpha=0.2)
        axes[1].plot(self.ells,np.median(ratio,axis=0),color='k', linestyle='-', linewidth=2)
        # uncertainties on difference
        ratio2 = (y_test - y_pred)#*(exponents[0]*np.prod(np.abs(X_test/theta_ref)**exponents[1:],axis=1))[:,None]
        a682, b682 = np.percentile(ratio2,percentile1,axis=0), np.percentile(ratio2,percentile2,axis=0)
        axes[2].fill_between(self.ells, b682, a682, color='k',alpha=0.2)
        axes[2].plot(self.ells,np.median(ratio2,axis=0),color='k', linestyle='-', linewidth=2)
            
        axes[0].scatter([], [], marker="+", color='k', s=30, label='True values')
        axes[0].plot([], [], color='k', lw=1., label='Recovered')
        
        axes[0].legend(loc='best', fontsize=15)
        axes[0].set_ylim(bottom=0)
        #axes[0].set_xlim(0, 1e4)
        axes[1].set_ylim(0.5,1.25)
        axes[2].set_ylim(-0.1,0.1)
        axes[1].axhline(.88, color='red')
        axes[1].axhline(1.12, color='red')
        axes[-1].set_xlabel(r"Angular multipole $\ell$", fontsize=15)
        axes[0].set_ylabel(r"$C_\ell^{kSZ}$ [$\mu$K$^2$]", fontsize=15)
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
    def __init__(self, dataset, hyperparameters, features, neurons=10, epochs=100,
                  scale_data=True, log_data=True, method='Neural Network', verbose=True):
        """First subclass with a unique implementation."""
        # Call the base class initializer to set up the common attributes
        super().__init__(features=features, dataset=dataset, hyperparameters=hyperparameters,
                          scale_data=scale_data, log_data=log_data, method=method, verbose=verbose)
        
        self.neurons = neurons
        self.epochs = epochs

    def regress(self, X_train, X_test, y_train, y_test):
        if self.verbose:
            print(f"Now running regression with {self.neurons} neurons in the middle layer and {self.epochs} epochs...")
            
        self.model = Sequential() #[Input(shape=input_shape), Dense(units=64, activation='relu')])
       # self.model.add(Dense(units=5, activation='relu')) # Input layer and first hidden layer (Dense layer)
        self.model.add(Dense(units=self.neurons, activation='leaky_relu')) # Second hidden layer
        self.model.add(Dense(units=self.neurons, activation='leaky_relu')) # Input layer and first hidden layer (Dense layer)
        self.model.add(Dense(units=self.neurons, activation='leaky_relu')) # Second hidden layer
        self.model.add(Dense(units=30, activation='linear'))      # Output layer (for regression, no activation function)
        
        # Compile the model
        self.model.compile(optimizer='RMSprop', loss='mean_squared_error')
        
        # Train the model
        self.history = self.model.fit(X_train, y_train,
                                                    epochs=self.epochs,
                                                    batch_size=32,
                                                    validation_data=(X_test, y_test),
                                                    verbose=0)
        return
                                    
        
    def metrics(self, X_test, y_test):
        # Evaluate the model
        loss = self.model.evaluate(X_test, y_test)
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
    
    # def save(self, path):
    #     # Save the Keras model
    #     self.model.save(f"{path}_model.keras")
    #     # Save the rest of the object with pickle
    #     with open(f"{path}_wrapper.pkl", "wb") as f:
    #         pickle.dump(self, f)

    # @staticmethod
    # def load(path):
    #     # Load the wrapper object first
    #     with open(f"{path}_wrapper.pkl", "rb") as f:
    #         wrapper = pickle.load(f)
    #     # Load the Keras model
    #     wrapper.model = tf.keras.models.load_model(f"{path}_model.keras")
    #     return wrapper


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
