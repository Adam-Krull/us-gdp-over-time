import keras
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from keras import layers
from sklearn.metrics import mean_squared_error

def read_files():
    '''Returns two dataframes from the csv files of information.'''
    historical = pd.read_csv('/kaggle/input/gdp-over-time-fr/historical.csv')
    recent = pd.read_csv('/kaggle/input/gdp-over-time-fr/recent.csv')
    #rename column for future transpose
    historical = historical.rename(columns={'Unnamed: 1': 'Year'})
    recent = recent.rename(columns={'Unnamed: 1': 'Year'})
    #strip whitespace from categories
    historical.Year = historical.Year.str.strip()
    recent.Year = recent.Year.str.strip()

    return historical, recent

def prepare_df(hist, rec):
    '''Takes in the two csv files.
    Returns a prepared dataframe for analysis.'''
    #useful variables for cleaning up the columns
    cols_of_interest = ['Private industries', 'Government', 'Agriculture, forestry, fishing, and hunting', 'Mining', 'Utilities',
                        'Construction', 'Manufacturing', 'Wholesale trade', 'Retail trade', 'Transportation and warehousing',
                        'Information', 'Data processing, internet publishing, and other information services',
                        'Finance and insurance', 'Real estate and rental and leasing', 'Professional and business services',
                        'Educational services', 'Health care and social assistance', 'Arts, entertainment, recreation, accommodation, and food services',
                        'Federal', 'State and local']
    rename = {'Agriculture, forestry, fishing, and hunting': 'Agriculture', 'Data processing, internet publishing, and other information services': 'Data services',
              'Real estate and rental and leasing': 'Real estate', 'Professional and business services': 'Professional services',
              'Health care and social assistance': 'Health care', 'Arts, entertainment, recreation, accommodation, and food services': 'Arts and entertainment'}
    #join the two dataframes together and clean up
    joined = hist.join(rec.set_index('Year'), how='left', on='Year', rsuffix='_recent')
    joined = joined.drop(columns=['Line', 'Line_recent', '1997_recent']).dropna()
    #transpose to make each category its own column
    joined = joined.set_index('Year').T
    #create subset of columns
    joined = joined[cols_of_interest]
    #shorten column names
    joined = joined.rename(columns=rename)
    #replace periods and cast as proper datatypes
    joined = joined.replace('...', 0.0).astype('float')
    joined.index = joined.index.astype('int')
    return joined

def eda_pipeline():
    '''Checks for filename of clean data. If it exists, reads it in.
    Otherwise, runs a simple pipeline to return the clean data.'''
    file = '/kaggle/input/gdp-over-time-fr/clean.csv'
    if os.path.isfile(file):
        return pd.read_csv(file, index_col=0)
    else:
        historical, recent = read_files()
        clean = prepare_df(historical, recent)
        clean.to_csv(file)
        return clean
    
def lineplot(df, categories, title):
    '''Takes in a number of categories and makes a line plot from them.'''  
    plt.figure(figsize=(9,6))
    for cat in categories:
        plt.plot(df.index, df[cat], label=cat)
    plt.legend()
    plt.xlabel('Year')
    plt.ylabel('Proportion of GDP Value')
    plt.title(title)
    plt.show()

def preprocess(df, category):
    '''Prepares a column of values for modeling.
    Takes a window of 10 values to predict the 11th.'''
    #provision empty arrays for data
    X = np.empty((66,10))
    y = np.empty((66,))
    #convert column from df to array
    arr = df[category].to_numpy()
    #iterate through to create slices of dataset
    for i in range(0, 66):
        X[i] = arr[i:i+10]
        y[i] = arr[i+10]
    #return full arrays (need them full, can provision later)    
    return X, y

def gen_baseline(X, y):
    '''Takes in X and y for one feature in the dataset.
    Returns the MSE for the predictions.'''
    #make prediction based on last X value in each set
    return round(mean_squared_error(y, X[:,9]), 4)

def gen_model():
    '''Creates and returns the best version of the GRU
    model I was able to produce.'''
    input_shape = (66,1)
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.GRU(16, activation='leaky_relu'),
        layers.Dense(8, activation='leaky_relu'),
        layers.Dense(1),
    ])
    return model

def train_model(model, X, y):
    '''Takes in the model and training data.
    Returns trained model.'''
    #define callback to stop training early
    callback = keras.callbacks.EarlyStopping(monitor='val_mean_squared_error', patience=3, verbose=1, restore_best_weights=True)
    #compile model
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
    #train model
    history = model.fit(X, y, batch_size=128, epochs=100, validation_split=0.2, verbose=0, callbacks=[callback])
    #return model
    return model, history

def best_mse(history):
    '''Takes in the history from model training to identify the best epoch.
    Best epoch determined based on lowest validate MSE.
    Returns lowest validate MSE and corresponding train MSE.'''
    #create series from val history
    val_series = pd.Series(history.history['val_mean_squared_error'])
    #get min value
    best_val = val_series.min()
    #get index (corresponds to epoch)
    idx = val_series.idxmin()
    #find matching mse in train dataset history
    best_train = history.history['mean_squared_error'][idx]
    return best_train, best_val

def make_preds(model, X):
    '''Takes in the trained model object and the X values.
    Projects out until 2030. Returns the predictioned values.'''
    #get pred for 2024 based on previous 10 years
    preds = []
    pred_2024 = model.predict(X[-1].reshape(1,-1))
    #add to list, need to index into value
    preds.append(pred_2024[0][0])
    #create empty array for more predictions
    X_new = np.empty((6,10))
    for i in range(1, 7):
        #set some values to known values from X
        X_new[i-1][:-i] = X[-1][i:]
        #set rest of values to the predictions
        X_new[i-1][-i:] = preds
        #predict again and add to predictions
        new_pred = model.predict(X_new[i-1].reshape(1,-1))[0][0]
        preds.append(new_pred)
    return preds   

def make_plot(df, category, preds):
    '''Takes in information about the specific category.
    Makes a plot of the known information and the predictions.'''
    plt.figure(figsize=(6,4))
    plt.plot(df.index, df[category], label='Historical')
    plt.plot(range(2024, 2031), preds, label='Predictions')
    plt.xlabel('Year')
    plt.ylabel('Proportion of GDP Value')
    plt.title(f'{category} predictions')
    plt.legend()
    plt.show()
    
def gen_preds(df):
    '''Takes in my dataframe of values. Goes through it one feature at a time.
    Preprocesses the data and trains a GRU model. Adds baseline MSE and validate
    MSE to an existing list of dictionaries. Predicts out through 2030 and plots
    the result. Pipeline function.'''
    #empty list for future info
    model_perf = []
    #iterate through features in df
    for col in df.columns:
        #create X and y
        X, y = preprocess(df, col)
        #generate baseline
        baseline = gen_baseline(X, y)
        #generate model
        model = gen_model()
        #train model
        model, history = train_model(model, X, y)
        #get metrics from best epoch
        train_mse, val_mse = best_mse(history)
        #create dictionary and add to list
        info_dict = {'Feature': col, 'Baseline': baseline, 'Train MSE': train_mse, 'Validate MSE': val_mse}
        model_perf.append(info_dict)
        #generate preds through 2030
        preds = make_preds(model, X)
        #make plot of predictions through 2030
        make_plot(df, col, preds)

    return pd.DataFrame(model_perf)