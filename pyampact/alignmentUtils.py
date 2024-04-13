"""
alignmentUtils
==============


.. autosummary::
    :toctree: generated/

    dp
    fill_priormat_gauss
    gh
    flatTopGaussIdx
    g
    flatTopGaussian
    viterbi_path
    mixgauss_prob
    fill_trans_mat
    orio_simmx
    simmx
    maptimes
    calculate_f0_est
    f0_est_weighted_sum
    f0_est_weighted_sum_spec
"""

import numpy as np
import pandas as pd
import librosa
import warnings

from scipy.signal import gaussian
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture

import sys

__all__ = [
    "dp",
    "fill_priormat_gauss",
    "gh",
    "flatTopGaussIdx",
    "g",
    "flatTopGaussian",
    "viterbi_path",
    "mixgauss_prob",
    "fill_trans_mat",
    "orio_simmx",
    "simmx",
    "maptimes",
    "calculate_f0_est",
    "f0_est_weighted_sum",
    "f0_est_weighted_sum_spec"
]


def dp(M):
    """
    Use dynamic programming to find a min-cost path through matrix M.
    Return state sequence in p,q.

    :param M: 2D numpy array, the input matrix.

    :return: A tuple containing three elements:
        - p: List, the sequence of row indices for the min-cost path.
        - q: List, the sequence of column indices for the min-cost path.
        - D: 2D numpy array, the cost matrix.
    """    
    r, c = M.shape

    # Initialize cost matrix D
    D = np.zeros((r + 1, c + 1))
    D[0, :] = np.inf  # Use np.inf instead of np.nan for proper comparison
    D[:, 0] = np.inf
    D[0, 0] = 0
    D[1:, 1:] = M

    traceback = np.zeros((r + 1, c + 1), dtype=int)

    # Dynamic programming loop
    for i in range(1, r + 1):
        for j in range(1, c + 1):
            min_cost = min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
            if min_cost == D[i - 1, j]:
                direction = 1  # Move upward
            elif min_cost == D[i, j - 1]:
                direction = 2  # Move leftward
            else:
                direction = 0  # Move diagonally

            D[i, j] += min_cost
            traceback[i, j] = direction

    # Traceback from bottom right
    i, j = r, c
    p, q = [], []

    while i > 1 and j > 1:  # Adjust for 0-based indexing
        direction = traceback[i, j]
        if direction == 0:
            i -= 1
            j -= 1
        elif direction == 1:
            i -= 1
        else:
            j -= 1
        p.insert(0, i - 1)  # Adjust for 0-based indexing
        q.insert(0, j - 1)  # Adjust for 0-based indexing

    return p, q, D[1:, 1:]  # Return the corrected sequences and the trimmed D matrix


# Gaussian/Viterbi functions
def fill_priormat_gauss(Nobs, ons, offs, Nstates):
    """
    Creates a prior matrix based on the DTW alignment (supplied by the input
    variables ons and offs. A rectangular window with half a Gaussian on
    each side over the onsets and offsets estimated by the DTW alignment.
    
    :params Nobs: Number of observations
    :params ons: Vector of onset times predicted by DTW alignment
    :params offs: Vector of offset times predicted by DTW alignment
    :params Nstates: Number of states in the hidden Markov model

    :return prior: Prior matrix based on DTW alignment
    """
    if Nstates is None:
        Nstates = 5

    Nnotes = len(ons)
    prior = np.zeros((Nnotes * (Nstates - 1) + 1, Nobs))
    frames = np.arange(1, Nobs + 1)

    for i in range(Nnotes):
        row = (i - 1) * (Nstates - 1)
        insert = Nstates - 5

        # Silence
        prior[row + 1, :] = flatTopGaussian(frames, gh(ons, i - 1, offs, i - 1, frames, 0.5),
                                            g(offs, i - 1, frames), g(ons, i, frames), gh(ons, i, offs, i, frames, 0.5))
    
        # Throws Value Error for negative values, both here and in MATLAB
        prior[row + 2:row + 2 + insert - 1, :] = np.tile(prior[row + 1, :], (insert, 1))

        # Transient, steady state, transient
        prior[row + 2 + insert, :] = flatTopGaussian(frames, g(offs, i - 1, frames),
                                                     gh(offs, i - 1, ons, i, frames, 0.75),
                                                     gh(ons, i, offs, i, frames, 0.25), g(offs, i, frames))
        prior[row + 3 + insert, :] = flatTopGaussian(frames, g(offs, i - 1, frames),
                                                     g(ons, i, frames), g(offs, i, frames), g(ons, i + 1, frames))
        prior[row + 4 + insert, :] = flatTopGaussian(frames, g(ons, i, frames),
                                                     gh(ons, i, offs, i, frames, 0.75),
                                                     gh(offs, i, ons, i + 1, frames, 0.25), g(ons, i + 1, frames))

    # The last silence
    i += 1
    prior[row + 5 + insert, :] = flatTopGaussIdx(frames, ons, i - 1, offs, i - 1, offs, i, ons, i + 1)

    return prior


def gh(v1, i1, v2, i2, domain, frac=0.5):
    """
    Get an element that is `frac` fraction of the way between `v1[i1]` and `v2[i2]`, 
    but check bounds on both vectors. `frac` of 0 returns `v1[i1]`, `frac` of 1 returns `v2[i2]`, 
    `frac` of 0.5 (the default) returns halfway between them.

    :param v1: 1D numpy array, the first vector.
    :param i1: Integer, the index in `v1`.
    :param v2: 1D numpy array, the second vector.
    :param i2: Integer, the index in `v2`.
    :param domain: Domain to check bounds on both vectors.
    :param frac: Float, the fraction of the way between `v1[i1]` and `v2[i2]`. Default is 0.5.

    :return: Integer, the element that is `frac` fraction of the way between `v1[i1]` and `v2[i2]`.
    """
    x1 = g(v1, i1, domain)
    x2 = g(v2, i2, domain)
    return int(frac * x1 + (1 - frac) * x2)
 

def flatTopGaussIdx(x, b1, bi1, t1, ti1, t2, ti2, b2, bi2):
    """
    Create a window function that is zeros, going up to 1s with the left
    half of a gaussian, then ones, then going back down to zeros with
    the right half of another gaussian.

    :param x: 1D numpy array, the set of points over which this is to be calculated.
    :param b1: Float, the x coordinate 2 stddevs out from the mean of the first gaussian.
    :param bi1: Integer, the index in `b1`.
    :param t1: Float, the x coordinate of the mean of the first gaussian.
    :param ti1: Integer, the index in `t1`.
    :param t2: Float, the x coordinate of the mean of the second gaussian.
    :param ti2: Integer, the index in `t2`.
    :param b2: Float, the x coordinate 2 stddevs out from the mean of the second gaussian.
    :param bi2: Integer, the index in `b2`.

    :return: 1D numpy array, the window function.

    The points should be in that order. Vectors are indexed intelligently, 
    so you don't have to worry about overflows or underflows.
    """
    b1 = g(b1, bi1, x)
    t1 = g(t1, ti1, x)
    t2 = g(t2, ti2, x)
    b2 = g(b2, bi2, x)
    
    return flatTopGaussian(x, b1, t1, t2, b2)


def g(vec, idx, domain):
    """
    Get an element from `vec`, checking bounds. `Domain` is the set of points
    that `vec` is a subset of.

    :param vec: 1D numpy array, the input vector.
    :param idx: Integer, the index in `vec`.
    :param domain: 1D numpy array, the set of points that `vec` is a subset of.

    :return: The element from `vec` at index `idx` if `idx` is within the bounds of `vec`, 
            otherwise the first element of `domain` if `idx` is less than 1, 
            or the last element of `domain` if `idx` is greater than the length of `vec`.
    """
    if idx < 1:
        return domain[0]
    elif idx > len(vec):
        return domain[-1]
    else:
        return vec[idx - 1]


def flatTopGaussian(x, b1, t1, t2, b2):
    """
    Create a window function that is zeros, going up to 1s with the left 
    half of a gaussian, then ones, then going back down to zeros with the 
    right half of another gaussian. 

    :param x: 1D numpy array, the set of points over which this is to be calculated.
    :param b1: Float, the x coordinate 2 stddevs out from the mean of the first gaussian.
    :param t1: Float, the x coordinate of the mean of the first gaussian.
    :param t2: Float, the x coordinate of the mean of the second gaussian.
    :param b2: Float, the x coordinate 2 stddevs out from the mean of the second gaussian.

    :return: 1D numpy array, the window function.

    The points should be in that order. If any of [b1, t1, t2] > any of [t1, t2, b2], 
    a message 'Endpoints are not in order: b1, t1, t2, b2' will be printed.
    """
    if any([b1, t1, t2]) > any([t1, t2, b2]):
        print('Endpoints are not in order: ', b1, t1, t2, b2)

    def custom_normalize(arr):
        return arr / np.max(np.abs(arr))

    def custom_gaussian(x, std):
        win = gaussian(2 * int(4 * std) + 1, std)
        return np.convolve(x, win, mode='same')

    def custom_gaussian_filter(x, t1, t2, b1, b2):
        left_std = (t1 - b1) / 2 + 1
        middle = np.ones(t2 - t1 - 1)
        right_std = (b2 - t2) / 2 + 1

        left = custom_normalize(custom_gaussian(x, left_std))
        right = custom_normalize(custom_gaussian(x, right_std))

        takeOneOut = t1 == t2
        w = np.concatenate((left[0:t1], middle, right[t2 + takeOneOut:]))
        
        return w


def viterbi_path(prior, transmat, obslik):
    """
    VITERBI Find the most-probable (Viterbi) path through the HMM state trellis.
    path = viterbi(prior, transmat, obslik)

    :param prior(i): Pr(Q(1) = i)
    :param transmat(i,j): Pr(Q(t+1)=j | Q(t)=i)
    :param obslik(i,t): Pr(y(t) | Q(t)=i)

    :returns:
        - path(t): q(t), where q1 ... qT is the argmax of the above expression.
        - delta(j,t) = prob. of the best sequence of length t-1 and then going to state j, and O(1:t)
        - psi(j,t) = the best predecessor state, given that we ended up in state j at t
    """ 
    T = obslik.shape[1]    
    prior = prior.reshape(-1, 1)
    Q = len(prior)

    scaled = False
    delta = np.zeros((Q, T))    
    psi = np.zeros((Q, T), dtype=int)
    path = np.zeros(T, dtype=int)
    scale = np.ones(T)

    t = 0
        
    delta[:, t] = prior.flatten() * obslik[:, t]        

    if scaled:
        delta[:, t] /= np.sum(delta[:, t])
        scale[t] = 1 / np.sum(delta[:, t])

    psi[:, t] = 0    
    for t in range(1, T):
        for j in range(Q):            
            delta[j, t] = np.max(delta[:, t - 1] * transmat[:, j])
            delta[j, t] *= obslik[j, t]

        if scaled:
            delta[:, t] /= np.sum(delta[:, t])
            scale[t] = 1 / np.sum(delta[:, t])

    p, path[T - 1] = np.max(delta[:, T - 1]), np.argmax(delta[:, T - 1])

    for t in range(T - 2, -1, -1):
        path[t] = psi[path[t + 1], t + 1]

    return path


def mixgauss_prob(data, means, covariances, weights):
    # TODO: This docstring no longer corresponds to the function signature.
    # """
    # Notation: Y is observation, M is mixture component, and both may be conditioned on Q.
    # If Q does not exist, ignore references to Q=j below.
    # Alternatively, you may ignore M if this is a conditional Gaussian.

    # :param data(:,t): t'th observation vector     
    # :param mu(:,k): E[Y(t) | M(t)=k] 
    #     or mu(:,j,k) = E[Y(t) | Q(t)=j, M(t)=k]    
    # :param Sigma(:,:,j,k): Cov[Y(t) | Q(t)=j, M(t)=k]
    #     or there are various faster, special cases:
    #     - Sigma() - scalar, spherical covariance independent of M,Q.
    #     - Sigma(:,:) diag or full, tied params independent of M,Q. 
    #     - Sigma(:,:,j) tied params independent of M. 

    # :param mixmat(k): Pr(M(t)=k) = prior
    #     or mixmat(j,k) = Pr(M(t)=k | Q(t)=j) 
    #     Not needed if M is not defined.

    # :param unit_norm: - optional; if 1, means data(:,i) AND mu(:,i) each have unit norm (slightly faster)

    # :returns:
    #     - B(t) = Pr(y(t)) ||
    #     - B(i,t) = Pr(y(t) | Q(t)=i) 
    #     - B2(i,k,t) = Pr(y(t) | Q(t)=i, M(t)=k) 
    
    # If the number of mixture components differs depending on Q, just set the trailing
    # entries of mixmat to 0, e.g., 2 components if Q=1, 3 components if Q=2,
    # then set mixmat(1,3)=0. In this case, B2(1,3,:)=1.0.
    # """

    # Create a Gaussian Mixture Model
    gmm = GaussianMixture(n_components=2)  # Specify the number of components

    # Fit the GMM to your data
    gmm.fit(data)

    # Calculate the probabilities for each data point
    probs = gmm.predict_proba(data)    

    # 'probs' now contains the conditional probabilities for each data point and each component.
    N = len(data)
    K = len(means)    
    

    covariances = [np.eye(5), np.eye(5)]    
    likelihood_matrix = np.zeros((N, K))    

    
    for i in range(N):
        for j in range(K):
            likelihood = weights[j] * multivariate_normal.pdf(data[i], mean=means[j], cov=covariances[j])
            likelihood_matrix[i][j] = likelihood

    return likelihood_matrix


# Matrix functions
def fill_trans_mat(trans_seed, notes):
    """
    Makes a transition matrix from a seed transition matrix.  The seed
    matrix is composed of the states: steady state, transient, silence,
    transient, steady state, but the full transition matrix starts and
    ends with silence, so the seed with be chopped up on the ends.
    Notes is the number of times to repeat the seed.  Transseed's first
    and last states should be equivalent, as they will be overlapped
    with each other.
    
    :param trans_seed: Transition matrix seed.
    :param notes: Number of notes being aligned.
    
    :return trans: Transition matrix
    """

    # Set up transition matrix
    N = trans_seed.shape[0]    
    trans = np.zeros((notes * (N - 1) + 1, notes * (N - 1) + 1))
    Non2 = int(np.ceil(N / 2 + 1)) # ADDED ONE!
    

    # Fill in the first and last parts of the big matrix with the
    # appropriate fragments of the seed
    trans[0:Non2, 0:Non2] = trans_seed[Non2:, Non2:]
    # trans[1:Non2, 1:Non2] = trans_seed[Non2:, Non2:] # Changed 0 to 1 here
    trans[-Non2:, -Non2:] = trans_seed[0:Non2, 0:Non2]

    # Fill in the middle parts of the big matrix with the whole seed
    for i in range(Non2, (notes - 1) * (N - 1) + 1 - Non2 + 1, N - 1):
        trans[i:i + N, i:i + N] = trans_seed

    return trans


def orio_simmx(M, D):
    """
    Calculate an Orio&Schwartz-style (Peak Structure Distance) similarity matrix.

    :param M: Binary mask where each column corresponds to a row in the output matrix S.
    :param D: Regular spectrogram, where columns of S correspond to columns of D.

    :return: S, the similarity matrix.
    """
    
    # Convert to NumPy arrays if input is DataFrame
    M = M.values if isinstance(M, pd.DataFrame) else M
    D = D.values if isinstance(D, pd.DataFrame) else D

    # Ensure compatibility for matrix multiplication
    
    # if M.shape[1] != D.shape[0]:
    #     M = M.T  # Transpose M if the number of columns in M does not match the number of rows in D

    
    # Calculate the similarities
    S = np.zeros((M.shape[1], D.shape[1]))

    D = D**2
    M = M**2
    

    nDc = np.sqrt(np.sum(D, axis=0))
    # avoid div 0's
    nDc = nDc + (nDc == 0)

    # Evaluate one row at a time
    with warnings.catch_warnings(): # Suppress imaginary number warnings, for now
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        for r in range(M.shape[1]):
            S[r, :] = np.sqrt(M[:, r] @ D) / nDc

    return S


def simmx(A, B):
    """
    Calculate a similarity matrix between feature matrices A and B.

    :param A: The first feature matrix.
    :param B: The second feature matrix. If not provided, B will be set to A.

    :return: The similarity matrix between A and B.
    """
    A = A.values if isinstance(A, pd.DataFrame) else A
    B = B.values if isinstance(B, pd.DataFrame) else B

    EA = np.sqrt(np.sum(A**2, axis=0))
    EB = np.sqrt(np.sum(B**2, axis=0))

    M = (A.T @ B) / (EA[:, None] @ EB[None, :])

    return M


def maptimes(t, intime, outtime):    
    """
    Map the times in t according to the mapping that each point
    in intime corresponds to that value in outtime.

    :param t: 1D numpy array, input times.
    :param intime: 1D numpy array, input time points.
    :param outtime: 1D numpy array, output time points.

    :return: u, a 2D numpy array of mapped times.
    """    

    # Gives ons/offs from score
    tr, tc = t.shape  # Get the dimensions of t
    t_flat = t.flatten()  # Flatten t
    nt = t_flat.size  # Get the total number of elements in t_flat
    nr = len(intime)  # Get the length of intime
    u = t_flat.copy()  # Copy t_flat into u

    for i in range(nt):
        idx = min(np.argmax(intime > t_flat[i]), len(outtime) - 1)
        u[i] = outtime[idx]

    u = u.reshape(tr, tc)  # Reshape u to match the original dimensions of t

    return u

    
# Old function... March 16
def calculate_f0_est(filename, hop_length, win_ms, tsr):    
            
    y, sr = librosa.load(filename, sr=tsr)

    # Calculate the maximum absolute amplitude
    max_amplitude = np.max(np.abs(y))

    # Normalize the audio signal
    normalized_y = y / max_amplitude

    # Compute STFT
    stft = librosa.stft(normalized_y, n_fft=win_ms, hop_length=hop_length)

    # Compute magnitude and phase of STFT
    magnitude = np.abs(stft)
    phase = np.angle(stft)

    # Compute instantaneous frequency
    # Instantaneous frequency = Δ(phase) / Δ(time)
    # Compute phase difference between consecutive frames
    delta_phase = np.diff(phase, axis=1)
    # Compute time difference between consecutive frames
    delta_time = hop_length / tsr
    # Compute instantaneous frequency
    instantaneous_freq = np.diff(phase, axis=1) / delta_time

    # Estimate f0 by finding the dominant frequency bin at each time frame
    f0 = np.argmax(magnitude, axis=0) * tsr / win_ms

    # Estimate power at each time frame (sum of magnitude squared)
    power = np.sum(magnitude**2, axis=0)

    # Now you have f0 and power estimates for each time frame    
    time = np.arange(len(instantaneous_freq[0])) * hop_length / sr
    new_time_value = time[-1] + hop_length / sr
    time = np.append(time, new_time_value)

    return f0, power


# These are the new functions to replace calculate_f0_est
def f0_est_weighted_sum(x, f, f0i, fMax=5000, fThresh=None):
    """
    Calculate F0, power, and spectrum for an inputted spectral representation.

    :param x: FxT matrix of complex spectrogram values.
    :param f: FxT matrix of frequencies of each of those spectrogram values.
    :param f0i: 1xT vector of initial estimates of f0 for each time.
    :param fMax: Maximum frequency to consider in weighted sum.
    :param fThresh: Maximum distance in Hz from each harmonic to consider.

    :return: A tuple containing three elements:
        - f0: Vector of estimated f0s from noteStart_s to noteEnd_s.
        - p: Vector of corresponding "powers" from f0EstWeightedSum.
        - strips: Estimated spectrum for each partial.
    """
    # if ~exist('fMax', 'var') || isempty(fMax), fMax = 5000; end
    # if ~exist('fThresh', 'var') || isempty(fThresh), fThresh = 2*median(diff(f(:,1))); end
    # if fMax is None:
    #     fMax = 5000

    if fThresh is None:
        fThresh = 2 * np.nanmedian(np.diff(f[:, 0]))


    f[np.isnan(f)] = 0
    x2 = np.abs(x) ** 2
    np.isnan(x2)
    wNum = np.zeros_like(x2)
    wDen = np.zeros_like(x2)
    maxI = np.max(fMax / f0i[f0i > 0])
    strips = []

    for i in range(1, int(maxI) + 1):
        mask = np.abs(f - (f0i * i)) < fThresh
        strip = x2 * mask
        strips.append(strip)

        wNum += 1 / i * strip#.toarray()
        wDen += strip#.toarray()

    wNum *= (f < fMax)
    wDen *= (f < fMax)

    f0 = np.sum(wNum * f, axis=0) / np.sum(wDen, axis=0)
    pow = np.sum(wDen, axis=0)

    return f0, pow, strips


def f0_est_weighted_sum_spec(filename, noteStart_s, noteEnd_s, midiNote, y, sr, useIf=True):
    """
    Calculate F0, power, and spectrum for a single note.

    :param fileName: Name of wav file to analyze.
    :param noteStart_s: Start position (in seconds) of note to analyze.
    :param noteEnd_s: End position (in seconds) of note to analyze.
    :param noteMidi: Midi note number of note to analyze.
    :param useIf: If true, use instantaneous frequency, else spectrogram frequencies.

    :return: A tuple containing three elements:
        - f0: Vector of estimated f0s from noteStart_s to noteEnd_s.
        - p: Vector of corresponding "powers" from f0EstWeightedSum.
        - M: Estimated spectrum.
    """
    
    # Inputs:
    #   fileName     name of wav file to analyze
    #   noteStart_s  start position (in seconds) of note to analyze
    #   noteEnd_s    end position (in seconds) of note to analyze
    #   noteMidi     midi note number of note to analyze
    #   useIf        if true, use instantaneous frequency, else spectrogram frequencies
    #
    # Outputs:
    #   f0 - vector of estimated f0s from noteStart_s to noteEnd_s
    #   p - vector of corresponding "powers" from f0EstWeightedSum
    #   M - estimated spectrum

    # set
    nIter = 10

    # set window and hop
    win_s = 0.064
    win = round(win_s * sr)
    hop = round(win / 8)

    # load if gram
    freqs, times, D = librosa.reassigned_spectrogram(y=y, sr=sr, hop_length=hop)

   # indices for indexing into ifgram (D)
    noteStart_hop = int(np.floor(noteStart_s * sr / hop))
    noteEnd_hop = int(np.floor(noteEnd_s * sr / hop))
    inds = range(noteStart_hop, noteEnd_hop)
      

    x = np.abs(D[:, inds])**(1/6)

    f = np.arange(win/2 + 1) * sr / win

    if useIf:
        xf = freqs[:,inds]
    else:
        xf = np.tile(f, (x.shape[1], 1)).T

    f0i = librosa.midi_to_hz(midiNote)

    fMax = 5000
    fThresh = 2 * np.nanmedian(np.diff(xf[:, 0]))
    
    f0,_,_ = f0_est_weighted_sum(x, xf, f0i, fMax, fThresh)
    # tmp = f0
    # for _ in range(nIter):
    #     if ~np.isnan(f0).any():
    #       f0,_,_ = f0_est_weighted_sum(x, xf, f0)
    #       tmp = f0
    #     else:
    #       f0 = tmp

    _, p, partials = f0_est_weighted_sum(x ** 6, xf, f0,sr)


    M = partials[0]
    for i in range(1, len(partials)):
        M += partials[i]

    t = np.arange(len(inds)) * win_s

    return f0, p, t, M, xf
