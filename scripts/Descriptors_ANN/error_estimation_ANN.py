import numpy as np
import pandas as pd
import csv
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization 
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
pd.options.mode.chained_assignment = None

# Load dataset
df = pd.read_excel('Dataset_descriptors.xlsx', na_values=['na', 'nan'], index_col=0)
target_Hf = df['Hf(298K)']
target_S = df['S(298.15)']
target_C300 = df['C300']
target_C400 = df['C400']
target_C500 = df['C500']
target_C600 = df['C600']
target_C800 = df['C800']
target_C1000 = df['C1000']
target_C1500 = df['C1500']
features = df[df.columns[10:]]
input_dim = len(df.columns[10:])

def model2(input_dim, loss, r1, l1, r2, l2, r3, l3, r4, l4):

    model = Sequential()
    model.add(Dense(l1, input_dim=input_dim, kernel_initializer='he_uniform', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(r1))
    model.add(Dense(l2, kernel_initializer='he_uniform', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(r2))
    model.add(Dense(l3, kernel_initializer='he_uniform', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(r3))
    model.add(Dense(l4, kernel_initializer='he_uniform', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(r4))
    model.add(Dense(1, kernel_initializer='he_uniform'))
    model.compile(loss=loss, optimizer='adam')
    return model

# Open results file and write out headers
out_file = open("grid_search_ANN_descriptors.csv", 'w')
wr = csv.writer(out_file, dialect='excel')
headers = ['epochs', 'batch_size', 'loss', 'l1', 'l2', 'l3', 'l4', 'r1', 'r2', 'r3', 'r4',
           'error_ma', 'error_ms', 'error_rms', 'error_mp', 'error_max']
wr.writerow(headers)
out_file.flush()

for target in (target_Hf,target_S,target_C300,target_C400,target_C500,target_C600,target_C800,target_C1000,target_C1500):
    wr.writerow("--------")
    out_file.flush()
    
    # Define search space
    epochs = [1000, 2000, 3000]
    batch_size = [64]
    loss = ['mean_squared_error']
    l1 = [1024]
    l2 = [512]
    l3 = [256]
    l4 = [128]
    r1 = [0.05, 0.1]
    r2 = [0.05, 0.1]
    r3 = [0.05, 0.1]
    r4 = [0.05, 0.1]

    # Setup the grid to be searched over
    param_grid = dict(batch_size=batch_size, epochs=epochs, loss=loss, l1=l1, l2=l2, l3=l3, l4=l4, 
                      r1=r1, r2=r2, r3=r3, r4=r4, input_dim=[input_dim])

    # Make scikit-learn accepted Keras model
    model = KerasRegressor(build_fn=model2, verbose=2)

    # Define outer folds
    kFolds = KFold(n_splits=10, shuffle=True, random_state=1).split(X=features.values, y=target.values)

    # Define inner folds
    grid_search = GridSearchCV(model, param_grid, cv=KFold(n_splits=10, shuffle=True, random_state=1),
                               n_jobs=-1, verbose=0, scoring='neg_mean_squared_error')

    for index_train, index_test in kFolds:
        # Get train and test splits
        x_train, x_test = features.iloc[index_train].values, features.iloc[index_test].values
        y_train, y_test = target.iloc[index_train].values, target.iloc[index_test].values

        # Apply min max normalization
        scaler = MinMaxScaler().fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)

        # Fit
        grid_search.fit(x_train, y_train)

        # Get best params
        best_params = grid_search.best_params_

        # Calculate error metrics
        predictions = grid_search.predict(x_test)
        diff = y_test - predictions
        r2 = r2_score(y_test, predictions)
        error_ma = mean_absolute_error(y_test, predictions)
        error_ms = mean_squared_error(y_test, predictions)
        error_rms = np.sqrt(np.mean(np.square(diff)))
        error_mp = np.mean(abs(np.divide(diff, y_test))) * 100
        error_max = np.amax(np.absolute(diff))

        # Write results
        row = [best_params['epochs'], best_params['batch_size'], best_params['loss'],
               best_params['l1'], best_params['l2'], best_params['l3'], best_params['l4'],
	       best_params['r1'], best_params['r2'], best_params['r3'], best_params['r4'],
               r2, error_ma, error_ms, error_rms, error_mp, error_max]
        wr.writerow(row)
        out_file.flush()

out_file.close()