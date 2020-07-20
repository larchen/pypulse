import numpy as np

from scipy.fft import fft, fftfreq

def fspectrum(ts, pulse, N=0, remove_dc=True):
    """Computes the Fourier transform of a pulse.

    Args:
        ts (numpy.array): An array of floats specifying the time axis of the
            pulse.
        pulse (numpy.array): A complex array specifying the pulse.
        N (int): The number of additional timesteps to add to the pulse. This
            will append N zeros to the end of the pulse in order to increase
            the resolution of the fourier transform in frequency space.
        remove_dc (bool): If true, the DC offset (mean) will be subtracted from
            the pulse prior to computing the fourier transform.
    
    Returns:
        tuple: A tuple (ks, fs) representing the frequencies and the fourier 
            transform of the input pulse.
    """
    ks = fftfreq(len(ts) + N, ts[1] - ts[0])
    
    if N > 0:
        pulse = np.concatenate([pulse, pulse[-1]*np.ones(N)])
    if remove_dc:
        pulse -= np.mean(pulse)

    fs = fft(pulse)
    
    fs = fs[np.argsort(ks)]
    ks = np.sort(ks)
    return ks, fs

def upconvert(
    pulse,
    frequency,
    phase=0.0,
    sampling_rate=1.0e9,
    time_steps_per_sample=100, 
    interpolate=None
    ):
    """Upconverts a pulse by mixing the pulse envelope with a carrier signal.

    Args:
        pulse (np.array): A complex array specifying the pulse envelope for
            each sample. The real and imaginary components of the amplitude 
            specify the I and Q components respectively.
        frequency (float): The frequency (in Hz) of the carrier. The carrier
            is just a plane wave of the form $e^{2\pi i f + i\phi}
        phase (float): The phase in radians of the carrier. Defaults to `0.0`.
        sampling_rate (float): The sampling rate of the AWG, defines the 
            length of each sample. Defaults to 1.0e9 (1 GS/second)
        time_steps_per_sample (int): The number of timesteps to simulate for
            each sample. This defines the resolution of the simulation.
    Returns:
        tuple: A tuple (tlist, drive) of numpy arrays representing the list
            of time steps and the complex amplitude of the drive at each 
            time step.
    """

    sample_time = 1/sampling_rate
    
    N = len(pulse)
    T = N*sample_time
    
    tlist = np.arange(0, T, sample_time/time_steps_per_sample)

    # This compensates for rounding error
    tlist = tlist[:N*time_steps_per_sample]
    tlist = np.append(tlist, [T])

    if interpolate is None:
        envelope = np.repeat(pulse, time_steps_per_sample)
        envelope = np.append(envelope, [0])
    elif interpolate == 'linear':
        envelope = np.interp(
            tlist, np.linspace(0, T, N + 1), np.append(pulse, [0])
        )
    else:
        raise ValueError('Interpolation strategy not supported. Must be \'linear\' or None.')

    carrier = np.exp(1j*2*np.pi*frequency*tlist + 1j*phase)

    drive = 2*np.pi*carrier * envelope

    return tlist, drive

def drag(ts, pulse, lmbda, rescale=True):
    """Adds a first order DRAG correction to a simple pulse.

    This function will return a pulse that includes the first order DRAG correction.
    Assuming the pulse is purely real, the correction is calculated as 

    $$ Q = - \lambda\dot{I}$$

    where $\lambda$ is a DRAG coefficient with units $1/2\pi Hz^{-1}$. See [1].

    Args:
        ts (np.array): The time steps at which the pulse envelope is evaluated.
        pulse (np.array): The complex amplitude of the pulse at each time step.
            This should be purely real.
        lmbda (float): The DRAG coefficient.
        rescale (bool): If `true`, rescales the overal pulse such that the
            integrated pulse time of the DRAG pulse is the same as the original.

    Returns:
        np.array: A complex array of pulse amplitudes, where the imaginary
            component contains the first order DRAG correction.

    References:
        [1]: https://arxiv.org/abs/1809.04919.
        [2]: https://arxiv.org/abs/1904.06560.
    """

    delta_t = ts[1]-ts[0]

    drag_pulse = pulse + 1j * lmbda * np.gradient(pulse) / delta_t

    if rescale:
        drag_pulse = drag_pulse * np.sum(np.abs(pulse)) / np.sum(np.abs(drag_pulse))

    return drag_pulse

def square(length, amplitude=1):
    """Constructs a square pulse.

    Args:
        length (int): The total length (number of samples) of the pulse.
        amplitude (float): The amplitude of the pulse.
    
    Returns:
        np.array: An array of amplitudes specifying the pulse.
    """

    return amplitude*np.ones(np.round(length).astype(int))

def cosine_ramp(length, ramp_length, amplitude=1):
    """Constructs a square pulse with cosine ramps.

    Args:
        length (int): The total length (number of samples) of the pulse.
        ramp_length (int): The length (number of samples) of each ramp.
        amplitude (float): The amplitude of the pulse.

    Returns:
        np.array: An array of amplitudes specifying the pulse.
    """

    ts = np.arange(length)
    cosine_pulse = amplitude*np.piecewise(
        ts,
        [
            ts < ramp_length, 
            np.logical_and(ts >= ramp_length, ts < length - ramp_length), 
            ts >= length - ramp_length
        ],
        [
            lambda t: -0.5*(np.cos(np.pi/ramp_length*t) - 1),
            1,
            lambda t: -0.5*(np.cos(np.pi*(t - length)/ramp_length) - 1)
        ]
    )

    return np.abs(cosine_pulse)

def gaussian(length, sigma, amplitude):
    """Constructs a gaussian pulse.

    Args:
        length: The total length (number of samples) of the pulse.
        sigma: The standard deviation of the Gaussian envelope, in
            units of the sampling time.
        A: The amplitude of the pulse.

    Returns:
        np.array: An array of amplitudes specifying the pulse.
    """

    ts = np.arange(length)

    pulse = amplitude * np.exp(- 0.5*(ts - length/2)**2/sigma)

    pulse -= pulse[0]

    return np.abs(pulse)