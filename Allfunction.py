import inspect
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import IPython.display
import warnings

from typing import Tuple

from cued_sf2_lab.familiarisation import load_mat_img, plot_image
from cued_sf2_lab.simple_image_filtering import halfcos, convse

from cued_sf2_lab.laplacian_pyramid import quantise, bpp, rowint, rowint2, rowdec, rowdec2

from cued_sf2_lab.dct import colxfm, regroup, dct_ii

from cued_sf2_lab.lbt import pot_ii, dct_ii

from cued_sf2_lab.dwt import idwt

# Laplacian Pyramid______________________________________
def compression_ratio(X, list_of_highpass, lowpass, ref_step, step_size):
    output = 0
    for hp in list_of_highpass:
        # get bits
        output += bpp(quantise(hp, step_size))*hp.size
    output += bpp(quantise(lowpass, step_size))*lowpass.size

    return (bpp(quantise(X, ref_step))*X.size/output, output)

def py4encoder(X, h, layers):
    # your code here
    Y_list = []
    X_prev = X
    for n in range(layers):
        X_cur = rowdec(rowdec(X_prev, h).T , h).T
        Z = rowint(rowint(X_cur, 2*h).T , 2*h).T
        Y_list.append(X_prev - Z)
        X_prev = X_cur

    return Y_list, X_prev

def py4decoder(Y_list, X_top, h):
    low_pass = []
    for y in reversed(Y_list):
        if low_pass:
            low_pass.append(rowint(rowint(low_pass[-1], 2*h).T , 2*h).T + y)
        else:
            low_pass.append(rowint(rowint(X_top, 2*h).T , 2*h).T + y)

    return low_pass

def rms_error(step, Y_list, X_top, X, h):
    Y_out = []
    for Y in Y_list:
        Y_out.append(quantise(Y, step))
    
    Z_out = py4decoder(Y_out, quantise(X_top, step), h)
    
    return np.std(X - Z_out[-1])

def optimisation(X, h, layers, target_MSE, start, end, size):
    Y_list_i_layer, Xi = py4encoder(X, h, layers)
    step_sizes = np.linspace(start, end, size)
    lowest = float("inf")
    for s in step_sizes:
        diff = abs(rms_error(s, Y_list_i_layer, Xi, X, h) - target_MSE)
        if diff < lowest:
            lowest = diff
            output = s
    return output, lowest

def step_ratios(layers, h):
    X_test = np.zeros((256,256))
    output = []
    for l in range(1, layers + 2):
        Y_list, X_top = py4encoder(X_test, h, l)
        Y_list[-1][Y_list[-1].shape[0]//2][Y_list[-1].shape[0]//2] = 100
        Z_list = py4decoder(Y_list, X_top, h)
        output.append(np.sum(Z_list[-1]**2.0))
    impulse_size = np.array([1/np.sqrt(mse) for mse in output])
    ratios = impulse_size/impulse_size[0]
    return ratios

def rms_error_MSE(step_sizes, Y_list, X_top, X, h):
    Y_out = []
    for n in range(len(Y_list)):
        Y_out.append(quantise(Y_list[n], step_sizes[n]))
    
    Z_out = py4decoder(Y_out, quantise(X_top, step_sizes[-1]), h)
    
    return np.std(X - Z_out[-1])

def optimisation_MSE(X, h, layers, target_MSE, start, end, size, step_ratios):
    Y_list_i_layer, Xi = py4encoder(X, h, layers)
    step_sizes = np.linspace(start, end, size)
    lowest = float("inf")
    for s in step_sizes:
        diff = abs(rms_error_MSE(s*step_ratios, Y_list_i_layer, Xi, X, h) - target_MSE)
        if diff < lowest:
            lowest = diff
            output = s
    return output

def compression_ratio_MSE(X, Y_list, X_top, step_scalar, step_ratios, step_ref):
    output = 0
    for n in range(len(Y_list)):
        output += bpp(quantise(Y_list[n], step_scalar*step_ratios[n]))*Y_list[n].size
    output += bpp(quantise(X_top, step_scalar*step_ratios[-1]))*X_top.size
    return bpp(quantise(X, step_ref))*X.size/output


# DCT ____________________________
def dctbpp(Yr, N):
    # Your code here
    """
    Calculate the entropy in bits per element (or pixel) for re-grouped image Yr

    The entropy represents the number of bits per element to encode re-grouped image Yr
    assuming an ideal first-order entropy code.
    """
    bits = 0
    step = Yr.shape[0]//N
    for row in range(0, Yr.shape[0], step):
        for column in range(0, Yr.shape[0], step):
            Ys = Yr[row:row + step, column:column + step]
            bits += bpp(Ys)*Ys.size

    return bits

def optimisation_DCT(Y, C, target_MSE, start, stop, total):
    lowest = float("inf")
    output = None
    step_sizes = np.linspace(start, stop, total)
    for step in step_sizes:
        Yq = quantise(Y, step)
        Z_quantised = colxfm(colxfm(Yq.T, C.T).T, C.T)
        diff = np.abs(np.std(X-Z_quantised) - target_MSE)
        if diff < lowest:
            lowest = diff
            output = step
    return (output,lowest)

def compression_ratio_DCT(X, Y, N, ref_step, step_size):
    output = 0
    Yq= quantise(Y, step_size)
    Yr = regroup(Yq, N)/N
    output += dctbpp(Yr, N)
    
    Xq = quantise(X, ref_step)

    return bpp(Xq)*Xq.size/output


def optimal_value_DCT (X, t, ref_step):
    Ct = dct_ii(t)

    Yt = colxfm(colxfm(X, Ct).T, Ct).T

    step = optimisation_DCT(Yt, Ct, np.std(X - quantise(X, ref_step)), 1, 50, 300)[0]

    Ytq = quantise(Yt, step)
    ztq = colxfm(colxfm(Ytq.T, Ct.T).T, Ct.T)

    number_of_bits = dctbpp(regroup(Ytq, t)/t, t)
    compression_ratios = compression_ratio_DCT(X, Yt, t, ref_step, step)

    return number_of_bits, compression_ratios, ztq

def optimal_list (X, N, t, ref_step):
    compression_ratios = []
    number_of_bits = []

    optimal_compression_ratios = []
    optimal_step_size_list = []
    optimal_scale_list = []
    output = []
    target_MSE = np.std(X - quantise(X, ref_step))

    C = dct_ii(N)
    scale_factors = np.linspace(1,2,50)
    step_sizes = []
    for scale in scale_factors:
        step_sizes.append(optimisation_DCT(Yt, Ct, np.std(X - quantise(X, ref_step)), 1, 50, 300)[0])
    
    step_sizes = np.array(step_sizes)
    compression_ratios = np.zeros(step_sizes.shape[0])

    for i in range(step_sizes.shape[0]):
        compression_ratios[i] = compression_ratio_DCT(X, scale_factors[i], N, 17, step_sizes[i])
    optimal_index = np.argmax(compression_ratios)    
    optimal_step = step_sizes[optimal_index]
    optimal_scale = scale_factors[optimal_index]
    optimal_compression_ratios.append(compression_ratios[optimal_index])
    optimal_step_size_list.append(optimal_step)
    optimal_scale_list.append(optimal_scale)


    number_of_bits.append(dctbpp(regroup(Ytq, t)/t, t))
    compression_ratios.append(compression_ratio_DCT(X, Yt, t, ref_step, step))





    Ct = dct_ii(t)

    Yt = colxfm(colxfm(X, Ct).T, Ct).T

    Ytq = quantise(Yt, optimal_step)
    Ztq = colxfm(colxfm(Ytq.T, Ct.T).T, Ct.T)
    output.append(Ztq)

    return step_sizes, compression_ratios, optimal_compression_ratios, optimal_scale_list, optimal_step_size_list, output




def DCT(X, image, t, ref_step):
    X, cmaps_dict = image
    X.shape[0] = X.shape[0] - X.shape[0]//2
    X.shape[1] = X.shape[1] - X.shape[1]//2

    number_of_bits, compression_ratios, ztq = optimal_value_DCT(X, t, ref_step)

    return (number_of_bits, compression_ratios, ztq)


#  LBT___________________________________
def optimisation_LBT(X, N, target_MSE, scale_factor, start, end, total):
    C = dct_ii(N)
    Pf, Pr = pot_ii(N, scale_factor)
    t = np.s_[N//2:-N//2]

    Xp = X.copy()  
    Xp[t,:] = colxfm(Xp[t,:], Pf)
    Xp[:,t] = colxfm(Xp[:,t].T, Pf).T
    Y = colxfm(colxfm(Xp, C).T, C).T
    step_sizes = np.linspace(start, end, total)
    error = float("inf")
    for step in step_sizes:
        Yq = quantise(Y, step)
        Z = colxfm(colxfm(Yq.T, C.T).T, C.T)
        Zp = Z.copy()  
        Zp[:,t] = colxfm(Zp[:,t].T, Pr.T).T
        Zp[t,:] = colxfm(Zp[t,:], Pr.T)
        diff = np.abs(np.std(X - Zp) - target_MSE)
        if diff < error:
            error = diff
            output = step
    return output

def compression_ratio_LBT(X, scale_factor, N, ref_step, step_size):
    output = 0
    C = dct_ii(N)
    Pf, Pr = pot_ii(N, scale_factor)
    t = np.s_[N//2:-N//2]  
    Xp = X.copy()  
    Xp[t,:] = colxfm(Xp[t,:], Pf)
    Xp[:,t] = colxfm(Xp[:,t].T, Pf).T
    Y = colxfm(colxfm(Xp, C).T, C).T
    Yq= quantise(Y, step_size)
    Yr = regroup(Yq, N)/N
    output += dctbpp(Yr, 16)
    Xq = quantise(X, ref_step)
    output = bpp(Xq)*Xq.size/output
    return output

def optimal_value_LBT(X, N, ref_step):
    optimal_compression_ratios = []
    optimal_step_size_list = []
    optimal_scale_list = []
    output = []
    target_MSE = np.std(X - quantise(X, ref_step))

    C = dct_ii(N)
    scale_factors = np.linspace(1,2,50)
    step_sizes = []
    for scale in scale_factors:
        step_sizes.append(optimisation_LBT(X, N, target_MSE, scale, 1, 50 , 300))
    
    step_sizes = np.array(step_sizes)
    compression_ratios = np.zeros(step_sizes.shape[0])

    for i in range(step_sizes.shape[0]):
        compression_ratios[i] = compression_ratio_LBT(X, scale_factors[i], N, 17, step_sizes[i])
    optimal_index = np.argmax(compression_ratios)    
    optimal_step = step_sizes[optimal_index]
    optimal_scale = scale_factors[optimal_index]
    optimal_compression_ratios.append(compression_ratios[optimal_index])
    optimal_step_size_list.append(optimal_step)
    optimal_scale_list.append(optimal_scale)
    
    Pf, Pr = pot_ii(N, optimal_scale)
    t = np.s_[N//2:-N//2]  
    Xp = X.copy() 
    Xp[t,:] = colxfm(Xp[t,:], Pf)
    Xp[:,t] = colxfm(Xp[:,t].T, Pf).T
    Y = colxfm(colxfm(Xp, C).T, C).T

    Yq = quantise(Y, optimal_step)
    Z = colxfm(colxfm(Y.T, C.T).T, C.T)
    Zp = Z.copy()  
    Zp[:,t] = colxfm(Zp[:,t].T, Pr.T).T
    Zp[t,:] = colxfm(Zp[t,:], Pr.T)
    output.append(Zp)

    return step_sizes, compression_ratios, optimal_compression_ratios, optimal_scale_list, optimal_step_size_list, output


def LBT(X, image, N, ref_step):
    X, cmaps_dict = image
    X.shape[0] = X.shape[0] - X.shape[0]//2
    X.shape[1] = X.shape[1] - X.shape[1]//2
    
    step_sizes, compression_ratios, optimal_compression_ratios, optimal_scale_list, optimal_step_size_list, output = optimal_value_LBT(X, N, ref_step)

    return (step_sizes, compression_ratios, optimal_compression_ratios, optimal_scale_list, optimal_step_size_list, output)




#  LBT ________________________________________________
def nlevdwt(X, n):
    m = X.shape[0]
    Y = X.copy()
    for _ in range(n):
        Y[:m, :m] = dwt(Y[:m, :m])
        m = m//2
    return Y

def nlevidwt(Y, n):
    m = Y.shape[0]//2**(n-1)
    Z = Y.copy()
    for _ in range(n):
        Z[:m , :m] = idwt(Z[:m, :m])
        m = m*2
    return Z

def quantdwt(Y: np.ndarray, dwtstep: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parameters:
        Y: the output of `dwt(X, n)`
        dwtstep: an array of shape `(3, n+1)`
    Returns:
        Yq: the quantized version of `Y`
        dwtenc: an array of shape `(3, n+1)` containing the entropies
    """
    # your code here
    n = dwtstep.shape[1] - 1
    dwtent = np.zeros(dwtstep.shape)
    m = Y.shape[0]
    Yq = Y.copy()
    for l in range(n):
        Yq[0:m//2, m//2: m] = quantise(Yq[0:m//2, m//2:m], dwtstep[0, l])
        dwtent[0,l] = bpp(Yq[0:m//2, m//2: m])*Yq[0:m//2, m//2: m].size
        
        Yq[m//2: m, 0:m//2] = quantise(Yq[m//2: m, 0:m//2], dwtstep[1, l])
        dwtent[1,l] = bpp(Yq[m//2: m, 0:m//2])*Yq[m//2: m, 0:m//2].size
        
        Yq[m//2: m, m//2: m] = quantise(Yq[m//2: m, m//2: m], dwtstep[2, l])
        dwtent[2,l] = bpp(Yq[m//2: m, m//2: m])*Yq[m//2: m, m//2: m].size
        m = m//2
                          
    Yq[0:m, 0:m] = quantise(Yq[0:m, 0:m], dwtstep[0,n])
    dwtent[0,n] = bpp(Yq[0:m, 0:m])*Yq[0:m, 0:m].size
    
    return Yq, dwtent

def compression_ratio_const_step(X, n, optimal_step, ref_step):
    const_dwtstep=np.array([[1 for _ in range(n + 1)], 
                               [1 for _ in range(n)] + [0], 
                               [1 for _ in range(n)] + [0]])
    Y = nlevdwt(X, n)
    const_dwtstep = const_dwtstep*optimal_step
    Yq, dwtent = quantdwt(Y, const_dwtstep)
    return bpp(quantise(X, ref_step))*X.size/np.sum(dwtent)

def optimisation_const_step(X, n, target_RMS, start, end, size):
    Y = nlevdwt(X, n)
    step_sizes = np.linspace(start, end, size)
    lowest = float("inf")
    
    for step in step_sizes:
        const_dwtstep=np.array([[1 for _ in range(n + 1)], 
                               [1 for _ in range(n)] + [0], 
                               [1 for _ in range(n)] + [0]])
        const_dwtstep = const_dwtstep*step
        Yq, dwtent = quantdwt(Y, const_dwtstep)
        Zq = nlevidwt(Yq, n)
        diff = abs(np.std(X - Zq) - target_RMS)
        if diff < lowest:
            lowest = diff
            output = step
        
    return output

def step_ratios(n):
    X_test = np.zeros((256, 256))
    dwt_ratios = np.ones((3, n + 1))
    m = 256
    for i in range(1, n + 1):
        Ytr = nlevdwt(X_test, i)
        Ytr[0:m//2, m//2: m][m//4][m//4] = 100
        Ztr = nlevidwt(Ytr, i)
        dwt_ratios[0][i-1] = np.sum(Ztr**2.0)
        
        Ybl = nlevdwt(X_test, i)
        Ybl[m//2: m, 0:m//2][m//4][m//4] = 100
        Zbl = nlevidwt(Ybl, i)
        dwt_ratios[1][i-1] = np.sum(Zbl**2.0)
        
        Ybr = nlevdwt(X_test, i)
        Ybr[m//2: m, m//2: m][m//4][m//4] = 100
        Zbr = nlevidwt(Ybr, i)
        dwt_ratios[2][i-1] = np.sum(Zbr**2.0)
        m = m//2
    Ytr = nlevdwt(X_test, n)
    Ytr[0:m, 0:m][m//2][m//2] = 100
    Ztr = nlevidwt(Ytr, n)
    dwt_ratios[0][n] = np.sum(Ztr**2.0)
    
    dwt_ratios = 1/np.sqrt(dwt_ratios)
    dwt_ratios = dwt_ratios/np.amax(dwt_ratios[0][0])
    dwt_ratios[-1][-1] = 0
    dwt_ratios[-2][-1] = 0

    return dwt_ratios


def LBT(X, image, h1, h2, n, ref_step, ):
    X, cmaps_dict = image
    X.shape[0] = X.shape[0] - X.shape[0]//2
    X.shape[1] = X.shape[1] - X.shape[1]//2

    UU = rowdec(U.T, h1).T
    UV = rowdec2(U.T, h2).T
    VU = rowdec(V.T, h1).T
    VV = rowdec2(V.T, h2).T

    # h1 = np.array([-1, 2, 6, 2, -1])/8
    # h2 = np.array([-1, 2, -1])/4

    # g1 = np.array([1, 2, 1])/2
    # g2 = np.array([-1, -2, 6, -2, -1])/4

    # Ur = rowint(UU.T, g1).T + rowint2(UV.T, g2).T
    # Vr = rowint(VU.T, g1).T + rowint2(VV.T, g2).T

    Y = nlevdwt(X, 4)
    Z = nlevidwt(Y, 4)

    target_RMS = np.std(X - quantise(X, 17))

    const_step_sizes = []
    compression_ratios_const_step = []

    const_step_sizes.append(optimisation_const_step(X, n, target_RMS, 1, 50, 200))
    compression_ratios_const_step.append(compression_ratio_const_step(X, n, const_step_sizes[n-1], ref_step))

    for n in list_to_inv:
    Yb_step = nlevdwt(X, n)
    const_dwtstep=np.array([[1 for _ in range(n + 1)], 
                               [1 for _ in range(n)] + [0], 
                               [1 for _ in range(n)] + [0]])
    const_dwtstep = const_dwtstep*const_step_sizes[n-1]
    Yqb_const, dwtentb_const = quantdwt(Yb_step, const_dwtstep)
    Zb_const = nlevidwt(Yqb_const, n)
    const_step_output.append(Zb_const)



    Yb_mse = nlevdwt(X, n)
    mse_dwtstep = mse_step_scalars[n-1]*step_ratios(n)
    Yqb_mse, dwtentb_mse = quantdwt(Yb_mse, mse_dwtstep)
    Zb_mse = nlevidwt(Yqb_mse, n)
    mse_output.append(Zb_mse)
    return