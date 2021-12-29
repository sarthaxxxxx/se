import numpy as np
import scipy.io.wavfile as wav


def _seg_signal_(signal, fs, window, tt_max):
    """
    Segment signal into chunks of length tt_max, w/ window_length = (window * fs).

    Parameters 
    ----------
    signal: numpy.ndarray
        Input speech signal
    fs: int
        Sampling frequency
    window: float
        window length in seconds
    tt_max: float
        chunk size (0 to 1)
    
    Returns
    -------
    segments: ndarray
        List of speech segments (includes silence)
    """
    assert signal.ndim == 2, "INVALID SIGNAL DIMENSIONS!!!"
    window_length, tt_max, segments = int(window * fs), int(tt_max * fs), []
    for idx in range(0, len(signal), window_length):
        beg_i, end_i = idx, idx + window_length
        chunk = signal[beg_i : end_i] if end_i < len(signal) else np.vstack([signal[beg_i : len(signal)] , np.zeros((end_i - len(signal), 1))])
        if end_i - beg_i > tt_max:
            s = 0; en = tt_max
            while en < len(chunk):
                segments.append(chunk[s : en])   
                if len(chunk) - en < tt_max:
                    pad_zeros = np.zeros(tt_max - (len(chunk) - en))
                    segments.append(np.vstack((chunk[en : ], np.zeros((pad_zeros.shape[0], 1)))))
                s += tt_max
                en += tt_max
        else:
            segments.append(np.vstack((chunk, np.zeros((tt_max - len(chunk), 1)))))
    return segments


def _read_slice_(path, window, tt_max):
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
    return _seg_signal_(_preemph_(norm_signal).reshape(-1, 1), fs, window, tt_max)


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



if __name__ == '__main__':
    sig = '/media/sarthak/Data/SER/data/clean_trainset/p226_002.wav'
    _read_slice_(sig, 1, 0.14)
