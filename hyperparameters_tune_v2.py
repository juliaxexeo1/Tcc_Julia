import numpy as np
from sktime.datasets import load_from_tsfile_to_dataframe
from sktime.datatypes._panel._convert import from_nested_to_2d_array
from sklearn.model_selection import cross_validate, RepeatedKFold
import lib_all_data_analysis_v3 as lda
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import RidgeClassifierCV
from itertools import combinations
import pywt



#Functions######################
def smooth_with_wavelets(y, wavename):
    #FUNCTION TO SMOOTH SIGNAL VIA WAVELET DECOMPOSITION
    #INPUTS:  - y = array-like signal to smooth
    #OUTPUTS: - y_rec = smoothed version of input signal

    # wavelet decomposition
    coeffs = pywt.wavedec(y, wavename, mode='symmetric')

    #zero out last X detail coeffs
    for i in range(len(coeffs) - 1):
        coeffs[i + 1] = np.zeros(coeffs[i + 1].shape)

    # wavelet recomposition rigth shift ONE
    y_rec = pywt.waverec(coeffs, wavename, mode='symmetric')

    return y_rec, coeffs

def calculate_error(sig_original, sig_smooth):
    rmse = np.sqrt(mean_squared_error(sig_original, sig_smooth))

    return rmse

def generate_power_set(A):
    power_set = []
    for r in range(1, len(A) + 1):
        subsets = combinations(A, r)
        power_set.extend(subsets)
    return power_set

def generate_power_set_min(A, min_elements=1):
    power_set = []
    for r in range(min_elements, len(A) + 1):
        subsets = combinations(A, r)
        power_set.extend(subsets)
    return power_set

def generate_subsets(iterable, subset_size):
    subsets = list(combinations(iterable, subset_size))
    return subsets
def evaluate_models(dataset, a_values, b_values, c_values, d_values):
    best_score = []
    best_cfg = []
    for a in a_values:
        for b in b_values:
            for c in c_values:
                for d in d_values:
                    #order = (a, b, c, d)
                    order = (a, b, c)
                    #Call WALE-a
                    X_transform = lda.wave_layer_ts_v2(X_accX_tab, a,wl_name=b, t_pool=c)
                    scores = cross_validate(clf_uni, X_transform, y_tab, cv=cv_STF, n_jobs=-1)
                    print('Conf. and Accuracy', order, scores['test_score'].mean())
                    best_score.append(scores['test_score'].mean())
                    best_cfg.append(order)

    return best_score, best_cfg
##################################

#Set repetions and cross-val
num_runs = 1
cross_val = 10

#Classifiers used
clf_uni = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
#clf_uni = RandomForestClassifier(n_estimators=50, criterion='gini', n_jobs=-1)
cv_STF = RepeatedKFold(n_splits=cross_val, n_repeats=num_runs, random_state=1)

# -- read data -------------------------------------------------------------
print('Loading data.....')
X, y = load_from_tsfile_to_dataframe('Earthquakes/Earthquakes.ts')

#Select one feature (accelerometer X signal)
X_accX = X.iloc[:, 0].to_frame()

#Convert nested dataset to tabular dataset
X_accX_tab = from_nested_to_2d_array(X_accX)

#Convert Dataframe to Array
X_accX_tab=X_accX_tab.to_numpy(dtype=np.float64)
y_tab=y.astype(int)


# -- run -------------------------------------------------------------------
print('Performing runs.....')

#Set generate
set_A = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14}
set_B = {0,1,2,3,4}
set_C = {0,1,2,3,4,5}
set_D = {'bior2.2', 'bior1.5', 'bior3.1', 'coif6', 'coif4', 'coif7', 'db1', 'db11', 'sym6', 'sym3', 'sym5', 'sym8', 'rbio4.4', 'rbio3.5', 'rbio3.3'}
set_E = ['bior1.1', 'bior1.5', 'bior1.5', 'bior2.2', 'bior2.4', 'bior2.6', 'bior2.8', 'bior3.1', 'bior3.3', 'bior3.5', 'bior3.7', 'bior3.9', 'bior4.4', 'bior5.5', 'bior6.8', 'coif1', 'coif2', 'coif3', 'coif4', 'coif5', 'coif6', 'coif7', 'coif8', 'coif9', 'coif10', 'coif11', 'coif12', 'coif13', 'coif14', 'coif15', 'coif16', 'coif17', 'db1', 'db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8', 'db9', 'db10', 'db11', 'db12', 'db13', 'db14', 'db15', 'db16', 'db17', 'db18', 'db19', 'db20', 'db21', 'db22', 'db23', 'db24', 'db25', 'db26', 'db27', 'db28', 'db29', 'db30', 'db31', 'db32', 'db33', 'db34', 'db35', 'db36', 'db37', 'db38', 'dmey', 'haar', 'rbio1.1', 'rbio1.3', 'rbio1.5', 'rbio2.2', 'rbio2.4', 'rbio2.6', 'rbio2.8', 'rbio3.1', 'rbio3.3', 'rbio3.5', 'rbio3.7', 'rbio3.9', 'rbio4.4', 'rbio5.5', 'rbio6.8', 'sym2', 'sym3', 'sym4', 'sym5', 'sym6', 'sym7', 'sym8', 'sym9', 'sym10', 'sym11', 'sym12', 'sym13', 'sym14', 'sym15', 'sym16', 'sym17', 'sym18', 'sym19', 'sym20']

#Set LENGHT WINDOW parameters
#a_values = [[0,1,3,5,6,7,9]]
a_values = [50,100,200,300]

#Set GROUPS of wavelets
#b_values = generate_subsets(set_D,3)
#Set WALE-a v1, v2 and v3
b_values = [['bior3.1', 'bior1.5', 'db11'],['coif6', 'sym6', 'bior2.2', 'coif4', 'db11', 'sym5'],['rbio4.4', 'sym5', 'bior3.1', 'coif6', 'sym8', 'coif7', 'bior2.2', 'db11', 'coif4', 'db1', 'rbio3.5', 'bior1.5']]

#Set POOLING parametres
c_values = [0,1,2]

#Not used
d_values = [0]
#d_values = [1,2]

#Instacie hyperparameters
scores, cfg = evaluate_models(X_accX_tab, a_values, b_values, c_values, d_values)
print('Best accuracy/error and CFG', scores[scores.index(max(scores))], cfg[scores.index(max(scores))])
maiores_valores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:15]
print('Os top 15')
for indice, valor in maiores_valores:
    print(f"Valor: {valor} | Índice: {cfg[indice]}")
