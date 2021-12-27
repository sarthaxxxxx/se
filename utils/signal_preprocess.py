import numpy as np
import scipy.io.wavfile as wav


def _seg_signal_(signal, fs, window, hop_per, mode):
    """
    Segment signal with window size (in ms) and hop_size = window_length * hop_per.

    Parameters 
    ----------
    signal: numpy.ndarray
        Input speech signal
    fs: int
        Sampling frequency
    window: float
        window length in seconds
    hop_per: float
        overlap percentage (0 to 1)
    mode: str
        Mode of data (train/test)
        If test, no overlap. 

    Returns
    -------
    segments: list
        List of speech segments (includes silence)
    """
    assert signal.ndim == 1, "Invalid signal dimension."
    window_length, segments = int(window * fs), []
    hop_length = int(hop_per * window_length) if mode == 'trainset' else window_length
    for idx in range(0, len(signal), hop_length):
        beg_i, end_i = idx, idx + window_length
        if end_i >= len(signal): 
            segments.append(np.append(signal[beg_i : len(signal)], [0] * (end_i - len(signal))))
            break
        segments.append(signal[beg_i : end_i])
    return np.array(segments, dtype = np.float64)


def _read_slice_(path, window, hop_per):
    """ Read input path to audio file and return normalised speech chunks.

    Parameters
    ----------
    path: string
        path to audio file
    window: float
        window length in seconds
    hop_per: float
        overlap percentage (0 to 1)

    Returns
    -------
        numpy.ndarray
        List of speech segments (includes silence)
    """
    fs, signal = wav.read(path)
    signal -= np.mean(signal, axis = 0, dtype = np.int16)
    C = 0.5 * np.sqrt(np.mean(np.square(signal), axis = 0))
    norm_signal = np.divide(signal, C)
    mode = path.split('/')[-2].split('_')[-1]
    return _seg_signal_(_preemph_(norm_signal), fs, window, hop_per, mode).reshape(-1, 1)


def _preemph_(signal, c = 0.95):
    """Apply pre-empahasis filter, of coefficient = 0.95 to train samples
    
    Parameters
    ----------
    signal: np.ndarray
        input signal 
    c: int
        coefficient of empahasis (0.95 fixed)

    Returns
    -------
        np.ndarray
        pre-emphasised input signal
    """
    s_0 = np.reshape(signal[0], [1, ])
    diff = signal[1:] - c * signal[:-1]
    return np.concatenate([s_0, diff], 0)


def _deemph_(signal_test, c = 0.95):
    """Apply de-empahasis filter, of coefficient = 0.95 to test sample outputs

    Parameters
    ----------
    y: np.ndarray
         signal 
    c: int
        coefficient of empahasis (0.95 fixed)

    Returns
    -------
    y: np.ndarray
        pre-emphasised test signal
    """
    y = np.zeros(signal_test.shape[0], dtype = np.float32)
    y[0] = signal_test[0]
    for idx in range(1, y.shape[0]):
        y[idx] = signal_test[idx] + c * (y[idx - 1])
    return y