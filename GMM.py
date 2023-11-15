import numpy as np
from scipy import stats
from sklearn.mixture import GaussianMixture
import pdb
#from morph_params import morph_params

np.random.seed(0)

'''
Reference code: https://github.com/VXU1230/Medium-Tutorials/blob/master/em/em.py
'''

def filter_using_std(x, mean, std, max_deviations):
        
    distance_from_mean = abs(x - mean) # MAE from class mean
        
    valid_pts = distance_from_mean < max_deviations * std
    
    #pdb.set_trace()
    return valid_pts 

def compute_statistics(array):
        
        min_value = np.min(array)
        max_value = np.max(array)
        mean = np.mean(array) #average
        std = np.std(array) #std deviation
        
        return min_value, max_value, mean, std

def learn_params(x, y, num_classes):
    '''
    code to learn pi, mu, sigma
    '''
    print('Computing parameters: pi, mu, sigma')

    N = len(x)
    
    pi = []
    mu = []
    sigma = []
    
    for k in range(num_classes):
        
        class_idx = np.where(y == k)[0]
        
        valence = x[:,0]
        arousal = x[:,1]

        min_valence, max_valence, avg_valence, std_valence = compute_statistics(valence[class_idx])
        min_arousal, max_arousal, avg_arousal, std_arousal = compute_statistics(arousal[class_idx])

        x_idx = filter_using_std(valence[class_idx], avg_valence, std_valence, max_deviations=1)
        y_idx = filter_using_std(arousal[class_idx], avg_arousal, std_arousal, max_deviations=1) 
        idx = class_idx[np.logical_and(x_idx, y_idx)]


        pi.append( len( x[class_idx]) /N ) #prior distribution N_k/N
        mu.append( np.mean( x[idx], axis=0) ) #class centroids
        sigma.append( np.cov( x[idx].T, bias= True) )

    return {'pi': pi, 'mu': mu, 'sigma': sigma}

def generate_GMM_labels(df, num_classes=8, merge=False):

    VA = np.array(df[["valence", "arousal"]])
    parameters = learn_params(VA, df['expression'], num_classes)
    
    weights = parameters['pi']
    means = parameters['mu']
    covariances = parameters['sigma']
    precision = [np.linalg.inv(mat) for mat in covariances]


    if merge == True:
        ### Merge with MorphSet mu and sigma ###
        print("Merging with Morphset Parameters")
        for k in range(len(morph_params['mu'])):
            parameters['mu'][k] = (parameters['mu'][k] + morph_params['mu'][k])/2
            parameters['sigma'][k] = (parameters['sigma'][k] + morph_params['sigma'][k])/2
        ###

    pdf = [stats.multivariate_normal( parameters["mu"][k], parameters["sigma"][k] ).pdf(VA) for k in range(num_classes)] #pdf for each k
    marginal = np.sum( [parameters['pi'][k] * pdf[k] for k in range(num_classes)], axis=0) #sum over all k
    
    var = (pdf/marginal).T * np.array(parameters['pi'])

    df['GMM_labels'] = [vector.tolist() for vector in var]

    return df, parameters

def run_EM(x, weights=None, means=None, covariances=None, tolerance=0.01):
    
    new_parameters = {} #store the new params

    #compute precision matrices from covariances
    precision = [np.linalg.inv(mat) for mat in covariances]

    model = GaussianMixture(n_components=8,
                            covariance_type='full',
                            tol=tolerance,
                            max_iter=1000,
                            weights_init=weights,
                            means_init=means,
                            precisions_init=precision)
    model.fit(x)

    new_parameters['pi'] = model.weights_
    new_parameters['mu'] = model.means_
    new_parameters['sigma'] = model.covariances_
    
    return model.predict(x), model.predict_proba(x), new_parameters

def generate_EM_labels(df, num_classes=8, tolerance=0.01, train_or_val="train"):

    VA = np.array(df[["valence", "arousal"]])
    parameters = learn_params(VA, df['expression'], num_classes)
    
    sklearn_forecasts, EM_soft_labels, new_params = run_EM(VA, parameters['pi'], parameters['mu'], parameters['sigma'], tolerance)

    df['GMM_labels'] = EM_soft_labels.tolist()
    '''
    if train_or_val == "train":
        df['expression'] = sklearn_forecasts # replace all with EM labels
    '''
    return df, new_params

