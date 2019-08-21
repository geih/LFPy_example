#!/usr/bin/env python
'''
Simple Hay simulation with one synaptic input, and extracellular electrodes
'''
from os.path import join
import numpy as np
import pylab as plt
import neuron
import LFPy
from hay_model.hay_active_declarations import active_declarations
nrn = neuron.h

def LFPy_ex(synaptic_y_pos=0, conductance_type='active', weight=0.01, input_spike_train=np.array([20.])):
    """
    Runs a NEURON simulation and returns an LFPy cell object for a single synaptic input.
    :param synaptic_y_pos: position along the apical dendrite where the synapse is inserted.
    :param conductance_type: Either 'active' or 'passive'. If 'active' all original ion-channels are included,
           if 'passive' they are all removed, yielding a passive cell model.
    :param weight: Strength of synaptic input.
    :param input_spike_train: Numpy array containing synaptic spike times
    :return: cell object where cell.imem gives transmembrane currents, cell.vmem gives membrane potentials.
             See LFPy documentation for more details and examples.
    """

    #  Making cell
    model_path = join('hay_model')
    neuron.load_mechanisms(join(model_path, 'mod'))
    cell_parameters = {
        'morphology': join(model_path, 'cell1.hoc'),
        'v_init': -65,
        'passive': False,
        'nsegs_method': 'lambda_f',
        'lambda_f': 100,
        'timeres_NEURON': 2**-4,  # Should be a power of 2
        'timeres_python': 2**-4,
        'tstartms': 0,
        'tstopms': 100,
        'custom_code': [join(model_path, 'custom_codes.hoc')],
        'custom_fun': [active_declarations],
        'custom_fun_args': [{'conductance_type': conductance_type}],
    }
    cell = LFPy.Cell(**cell_parameters)

    #  Making synapse
    synapse_parameters = {
        'idx': cell.get_closest_idx(x=0., y=synaptic_y_pos, z=0.),
        'e': 0.,
        'syntype': 'ExpSyn',
        'tau': 10.,
        'weight': weight,
        'record_current': True,
    }
    synapse = LFPy.Synapse(cell, **synapse_parameters)
    synapse.set_spike_times(input_spike_train)

    cell.simulate(rec_imem=True, rec_vmem=True)

    plot_electrode_signal(cell)
#    dense_2D_LFP(cell)


def plot_electrode_signal(cell):
    #  Making extracellular electrode
    elec_x = np.array([25.])
    elec_y = np.zeros(len(elec_x))
  #  elec_y = np.array([600.])
    elec_z = np.zeros(len(elec_x))

    electrode_parameters = {
        'sigma': 0.3,              # extracellular conductivity
        'x': elec_x,        # x,y,z-coordinates of contact points
        'y': elec_y,
        'z': elec_z,
    }
    electrode = LFPy.RecExtElectrode(cell, **electrode_parameters)
    electrode.calc_lfp()

    cell_plot_idx = 0
    plt.figure(127)
    plt.ion()    
    plt.subplots_adjust(hspace=0.3)  # Adjusts the vertical distance between panels.
    plt.subplot(132, aspect='equal')
    plt.axis('off')
    [plt.plot([cell.xstart[idx], cell.xend[idx]], [cell.ystart[idx], cell.yend[idx]], c='k') for idx in xrange(cell.totnsegs)]
    [plt.plot(electrode.x[idx], electrode.y[idx], 'bD') for idx in range(len(electrode.x))]
    plt.plot(cell.xmid[cell.synidx], cell.ymid[cell.synidx], 'y*', markersize=10)

    plt.subplot(231, title='Membrane potential', xlabel='Time [ms]', ylabel='mV')
    plt.plot(cell.tvec, cell.vmem[cell_plot_idx, :], color='k', lw=2) 
    plt.subplot(234, title='Transmembrane currents', xlabel='Time [ms]', ylabel='nA')
    plt.plot(cell.tvec, cell.imem[cell_plot_idx, :], color='k', lw=2)

    plt.seed(1234)
    #  Make signal with units uV (instead of mV) and add normally distributed white noise with RMS of 15
    signal_with_noise = 1000*electrode.LFP + np.random.normal(0, 15, size=electrode.LFP.shape)
#    signal_without_noise = 1000*electrode.LFP
    ylim = np.max(np.abs(signal_with_noise)) * 1.2
    plt.subplot(133, title='Extracellular potential', xlabel='Time [ms]', ylabel='$\mu V$', ylim=[-ylim, ylim])
    [plt.plot(cell.tvec, signal_with_noise[idx, :], c='b', lw=2) for idx in range(len(electrode.x))]
    plt.show()

def dense_2D_LFP(cell):

    #  Make dense 2D grid of electrodes
    x = np.linspace(-1000, 1000, 15)
    y = np.linspace(-500, 1500, 15)
    x, y = np.meshgrid(x, y)
    elec_x = x.flatten()
    elec_y = y.flatten()
    elec_z = np.zeros(len(elec_x))

    electrode_parameters = {
    'sigma': 0.3,              # extracellular conductivity
    'x': elec_x,        # x,y,z-coordinates of contact points
    'y': elec_y,
    'z': elec_z,
    }
    electrode = LFPy.RecExtElectrode(cell, **electrode_parameters)
    electrode.calc_lfp()
    plt.figure(128)
    plt.ion()
    [plt.plot([cell.xstart[idx], cell.xend[idx]], [cell.ystart[idx], cell.yend[idx]], c='k') for idx in xrange(cell.totnsegs)]
    plt.plot(cell.xmid[cell.synidx], cell.ymid[cell.synidx], 'y*', markersize=10)

    time_idx = np.argmax(cell.vmem[cell.synidx, :])
    sig_amp = 1000 * electrode.LFP[:, time_idx].reshape(x.shape)
    color_lim = np.max(np.abs(sig_amp))/5
    plt.imshow(sig_amp, origin='lower', extent=[np.min(x), np.max(x), np.min(y), np.max(y)],
               vmin=-color_lim, vmax=color_lim, interpolation='nearest', cmap=plt.cm.bwr)
    plt.colorbar()
    plt.show()

if __name__ == '__main__':
    LFPy_ex()
