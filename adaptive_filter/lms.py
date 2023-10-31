__authors__ = ['Joao Felipe Guedes da Silva <guedes.joaofelipe@poli.ufrj.br>']
__modifiers = ['Minxuan Qin']
import numpy as np 
from .utils import rolling_window
from scipy.fftpack import dct
from scipy.linalg import dft

class LMS:
    """ 
    Implements the Complex LMS algorithm for COMPLEX valued data.
        (Algorithm 3.2 - book: Adaptive Filtering: Algorithms and Practical
                                                       Implementation, Diniz)

    Base class for other LMS-based classes

    ...

    Attributes
    ----------    
    . step: (float)
        Convergence (relaxation) factor.
    . filter_order : (int)
        Order of the FIR filter.
    . init_coef : (row np.array)
        Initial filter coefficients.  (optional)
    . d: (row np.array)
        Desired signal. 
    . x: (row np.array)
        Signal fed into the adaptive filter. 
    
    Methods
    -------
    fit(d, x)
        Fits the coefficients according to desired and input signals
    
    predict(x)
        After fitted, predicts new outputs according to new input signal    
    """
    def __init__(self, step, filter_order, init_coef = None):        
        self.step = step
        self.filter_order = filter_order
        self.init_coef = np.array(init_coef)
    
        # Initialization Procedure
        self.n_coef = self.filter_order + 1
        self.d = None
        self.x = None
        self.n_iterations = None
        self.error_vector = None
        self.output_vector = None
        self.coef_vector = None
        
    def __str__(self):
        """ String formatter for the class"""
        return "LMS(step={}, filter_order={})".format(self.step, self.filter_order)
        
    def fit(self, d, x):
        """ Fits the LMS coefficients according to desired and input signals
        
        Arguments:
            d {np.array} -- desired signal
            x {np.array} -- input signal
        
        Returns:
            {np.array, np.array, np.array} -- output_vector, error_vector, coef_vector
        """
        # Pre allocations
        self.d = np.array(d)        
        self.x = np.array(x)        
        self.n_iterations = len(self.d)
        self.output_vector = np.zeros([self.n_iterations], dtype=complex)
        self.error_vector = np.zeros([self.n_iterations], dtype=complex)
        self.coef_vector = np.zeros([self.n_iterations+1, self.n_coef], dtype=complex)

        # Initial State Weight Vector if passed as argument        
        self.coef_vector[0] = np.array([self.init_coef]) if self.init_coef is not None else self.coef_vector[0]

        # Improve source code regularity
        prefixed_x = np.append(np.zeros([self.n_coef-1]), self.x)        
        X_tapped = rolling_window(prefixed_x, self.n_coef)

        for k in np.arange(self.n_iterations):                    
            regressor = X_tapped[k]

            self.output_vector[k] = np.dot(np.conj(self.coef_vector[k]), regressor)            
            self.error_vector[k] = self.d[k]-self.output_vector[k]
            self.coef_vector[k+1] = self.coef_vector[k]+self.step*np.conj(self.error_vector[k])*regressor
                        
        return self.output_vector, self.error_vector, self.coef_vector
    
    def predict(self, x):
        """ Makes predictions for a new signal after weights are fit
        
        Arguments:
            x {row np.array} -- new signal
        
        Returns:
            float -- resulting output""" 

        # taking the last n_coef iterations of x and making w^t.x
        return np.dot(self.coef_vector[-1], x[:-self.n_coef])


class NLMS(LMS):
    def __init__(self, step, filter_order, gamma, init_coef = None, p=0.1, f_0=0.2, hr_est=60):                
        LMS.__init__(self, step=step, filter_order=filter_order, init_coef=init_coef)        
        self.gamma = gamma
        ## MOD Minxuan: experimental value p & f_0, to make the step size between 0 and 1
        ## hr_est: estimated heart rate (already converted to the corresponding fs)
        self.p = p
        self.f_0 = f_0
        self.hrv = hr_est
        
    def __str__(self):
        ## MOD Minxuan: add corresponding parameter
        return "NLMS(step={}, gamma={}, filter_order={}, p_value={}, f_0={}, heart rate={})".format(self.step, self.gamma, self.filter_order, self.p, self.f_0, self.hrv)
    
    ## MOD Minxuan: change the step size based on SNR
    ## Help function for change_step
    def determine_step(self, prod_std):
        prod_std = prod_std / self.p
        if prod_std > 0.9:
            self.step = 0.9
        elif prod_std < 0.1:
            self.step = self.f_0
        else:
            self.step = prod_std
    

    ## It should be called in the fit function
    ## Assume total iteration is known, d & x as well
    def change_step(self, curr_time):
        ## curr_time is the 'k' in the fit function loop
        ## compute std of d & x in the last half heart rate cycle and pass it to determine_step
        sig_len = len(self.d)
        half_len = self.hrv // 2
        if curr_time >= half_len:
            d_std = np.std(self.d[curr_time-half_len:curr_time])
            x_std = np.std(self.x[curr_time-half_len:curr_time])
        elif curr_time == 0:
            d_std = self.d[curr_time]
            x_std = self.x[curr_time]
        else:
            d_std = np.std(self.d[:curr_time])
            x_std = np.std(self.x[:curr_time])
        prod_std = d_std * x_std
        self.determine_step(prod_std)

    def fit(self, d, x):
        # Pre allocations
        self.d = np.array(d)        
        self.x = np.array(x)        
        self.n_iterations = len(self.d)
        self.output_vector = np.zeros([self.n_iterations], dtype=complex)
        self.error_vector = np.zeros([self.n_iterations], dtype=complex)
        self.coef_vector = np.zeros([self.n_iterations+1, self.n_coef], dtype=complex)
        # Initial State Weight Vector if passed as argument        
        self.coef_vector[0] = np.array([self.init_coef]) if self.init_coef is not None else self.coef_vector[0]

        # Improve source code regularity
        prefixed_x = np.append(np.zeros([self.n_coef-1]), self.x)        
        X_tapped = rolling_window(prefixed_x, self.n_coef)

        for k in np.arange(self.n_iterations):                    
            regressor = X_tapped[k]

            self.output_vector[k] = np.dot(np.conj(self.coef_vector[k]), regressor)            
            self.error_vector[k] = self.d[k]-self.output_vector[k]
            ## MOD Minxuan: change the step size based on SNR
            self.change_step(k)          
            self.coef_vector[k+1] = self.coef_vector[k]+(self.step/(self.gamma+np.dot(np.conj(regressor.T), regressor)))*np.conj(self.error_vector[k])*regressor
            
        return self.output_vector, self.error_vector, self.coef_vector
