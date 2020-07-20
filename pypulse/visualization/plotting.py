"""This module provides a set of functions for visualizing simulation results.
"""

import numpy as np
import matplotlib as mpl
import itertools as it

import matplotlib.pyplot as plt

from pypulse.visualization.bloch import BlochSphere
from pypulse import fspectrum

def plot_pulse(ts, pulse, fourier_kwargs={}, IQ=True, **kwargs):
    """Plot a pulse.

    Args:
        ts (numpy.array): An array of floats specifying the time axis of the pulse.
        pulse (numpy.array): An array specifying the complex pulse amplitudes.
        fourier_kwargs (dict): A dictionary of keyword arguments to pass to the 
            `fspectrum` function.
        IQ (bool): If `True`, plots the in-phase (I) and quadrature (Q) components
            of the pulse. Otherwise, plots the amplitude and phase of the pulse.

    Returns:
        tuple: A tuple (fig, axes) representing the matplotlib figure and axes.
    """
    
    fig, axes = plt.subplots(3, 1, figsize=kwargs.setdefault('figsize', (7, 5)))
    ax = axes[0]

    if IQ:
        ax.step(ts, np.real(pulse), color='C0', where='post')
        ax.fill_between(ts, np.real(pulse), color='C0', step='post', alpha=0.2)
        ax.set_ylabel('I (Hz)')    
    else:
        ax.step(ts, np.abs(pulse), color='C0', where='post')
        ax.fill_between(ts, np.abs(pulse), color='C0', step='post', alpha=0.2)
        ax.set_ylabel('Amplitude (Hz)')
    ax.set_xlabel('Time (s)')
    
    ax = axes[1]

    if IQ:
        ax.step(ts, np.imag(pulse), color='C1', where='post')
        ax.fill_between(ts, np.imag(pulse), color='C1', step='post', alpha=0.2)
        ax.set_ylabel('Q (Hz)')
    else:
        ax.step(ts, np.angle(pulse), color='C1', where='post')
        ax.fill_between(ts, np.angle(pulse), color='C1', step='post', alpha=0.2)
        ax.set_ylabel('Phase (rad)')
    ax.set_xlabel('Time (s)')
    
    ax = axes[2]
    fourier_kwargs.setdefault('N', int(1e6))
    fourier_kwargs.setdefault('remove_dc', True)
    ks, fs = fspectrum(ts, pulse, **fourier_kwargs)
    if fourier_kwargs['remove_dc']:
        fs[len(fs)//2] = np.average(fs[[len(fs)//2-1, len(fs)//2+1]])
    ax.plot(ks, np.abs(fs)/np.max(np.abs(fs)), color='C2')
    ax.set_ylabel('FFT')
    ax.set_xlabel('Frequency (Hz)')
    
    fig.tight_layout()
    return fig, axes

def plot_fft(ts, pulse, fourier_kwargs={}, dB=True, flist=[], **kwargs):
    """Plots the FFT of a pulse.

    Args:
        ts (numpy.array): An array of floats specifying the time axis of the pulse.
        pulse (numpy.array): An array specifying the complex pulse amplitudes.
        fourier_kwargs (dict): A dictionary of keyword arguments to pass to the 
            `fspectrum` function.
        dB (bool): If true, sets the y-axis to a log scale.
        flist (list): A list of frequencies to plot along with the FFT.

    Returns:
        tuple: A tuple (fig, axes) representing the matplotlib figure and axis.
    """

    fourier_kwargs.setdefault('N', int(1e6))
    fourier_kwargs.setdefault('remove_dc', True)
    ks, fs = fspectrum(ts, pulse, **fourier_kwargs)
    if fourier_kwargs['remove_dc']:
        fs[len(fs)//2] = np.average(fs[[len(fs)//2 - 1, len(fs)//2 + 1]])
    
    fig, ax = plt.subplots(figsize=kwargs.setdefault('figsize', (5, 3)))
    ax.plot(ks, np.abs(fs)/np.max(np.abs(fs)), color='grey')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('FFT')

    for i, f in enumerate(flist):
        ax.axvline(f, color=f'C{i}')
    
    if dB:
        ax.set_yscale('log')
    fig.tight_layout()

    return fig, ax


def rotating_frame(t, x, y, z=0, omega=0):
    """Computes the x, y coordinates in a frame rotating with frequency omega.

    This function currently only supports rotations about the Z axis, but should
    be extended to support an arbitrary axis of rotation.

    Args:
        t (numpy.array): A list of time steps corresponding to the trajectory. 
        x (numpy.array): The X values at each time step.
        y (numpy.array): The Y values at each time step.
        z (numpy.array): The Z values at each time step. (Not used)
        omega (float): The frequency of rotation (in angular units).

    Returns:
        tuple: A tuple (x, y, z) of the coordinates in the rotating frame.
    """
    rot_x = x*np.cos(omega*t) - y*np.sin(omega*t)
    rot_y = y*np.cos(omega*t) + x*np.sin(omega*t)

    return rot_x, rot_y, z

def plot_bloch(t, x, y, z, omega=0.0, bloch=None, **kwargs):
    """Plots a trajectory on the bloch sphere.

    Args:
        t (numpy.array): A list of time steps corresponding to the trajectory. 
        x (numpy.array): The X values at each time step.
        y (numpy.array): The Y values at each time step.
        z (numpy.array): The Z values at each time step.
        omega (float): The frequency of the rotating frame (in angular units).
            Defaults to zero for a trajectory that is already in the proper
            frame.
        bloch (BlochSphere): An instance of bloch sphere to add the points to.
    
    Returns:
        BlochSphere: An instance of BlochSphere.
    """
    x, y, z = rotating_frame(t, x, y, z, omega)

    
    b = BlochSphere(**kwargs) if bloch is None else bloch
    
    b.add_points([x,y,z])
    return b
    
def plot_populations(t, populations, shape=None, **kwargs):
    """Plots the populations at each timestep.

    Args:
        t (numpy.array): A list of timesteps corresponding to the trajectory.
        populations (numpy.array): A 2D array where the 0-axis represents the 
            level and the 1-axis represents the populations at each time step.

    Returns:
        tuple: A tuple (fig, ax) representing the matplotlib figure and axis.
    """

    if shape is None:
        shape = (len(populations))

    states = [
        ''.join(p) for p in it.product(*[map(str, range(d)) for d in shape])
    ]

    fig, ax = plt.subplots(figsize=kwargs.pop('figsize', (5,3)))
    for i, p in enumerate(populations):
        ax.plot(t, p, f'C{i}', label=rf'$|{states[i]}\rangle$')
    
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Population')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    return fig, ax

def plot_paulis(t, x, y, z, omega=0.0, **kwargs):
    """Plots the pauli expectations at each timestep.

    Args:
        t (numpy.array): A list of timesteps corresponding to the trajectory.
        x (numpy.array): The X values at each time step.
        y (numpy.array): The Y values at each time step.
        z (numpy.array): The Z values at each time step.
        omega (float): The frequency of the rotating frame (in angular units).
            Defaults to zero for a trajectory that is already in the proper
            frame.

    Returns:
        tuple: A tuple (fig, ax) representing the matplotlib figure and axis.
    """
    x, y, z = rotating_frame(t, x, y, z, omega)

    fig, axes = plt.subplots(3, 1, figsize=kwargs.pop('figsize', (5,5)))
    ax = axes[0]
    ax.plot(t, x, 'C0')
    ax.set_ylabel(r'$\langle X\rangle$')
    ax.set_ylim(-1.05, 1.05)
    ax = axes[1]
    ax.plot(t, y, 'C1')
    ax.set_ylabel(r'$\langle Y\rangle$')
    ax.set_ylim(-1.05, 1.05)
    ax = axes[2]
    ax.plot(t, z, 'C2')
    ax.set_ylabel(r'$\langle Z\rangle$')
    ax.set_ylim(-1.05, 1.05)
    fig.tight_layout()

    return fig, axes