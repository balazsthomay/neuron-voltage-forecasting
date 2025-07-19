import numpy as np
from brian2 import *
import os

# code source: https://brian2.readthedocs.io/en/stable/examples/frompapers.Brunel_2000.html

# Set simulation parameters
sim_time = 10 * second  # Simulation duration set to 10 seconds
run_ids = range(3)     # Number of experimental runs
output_dir = 'output'   # Define subfolder name for saving files
os.makedirs(output_dir, exist_ok=True)  # Create output subfolder if it doesn't exist

def sim(sim_time, run_id):
    """
    sim_time -- simulation time (10 seconds)
    run_id -- identifier for each experimental run
    """
    seed(12345 + run_id)  # Set unique random seed for reproducibility

    # Network parameters (adjusted for project: 80 excitatory, 20 inhibitory neurons)
    N_E = 80  # Number of excitatory neurons
    N_I = 20  # Number of inhibitory neurons
    N = N_E + N_I
    epsilon = 0.1  # Connection probability as specified
    C_E = epsilon * N_E
    C_ext = C_E

    # Neuron parameters (adjusted for project specifications)
    tau = 10 * ms      # Membrane time constant set to 10 ms
    E_L = -70 * mV     # Resting potential set to -70 mV
    theta = -50 * mV   # Threshold potential set to -50 mV
    V_r = -70 * mV     # Reset potential set to -70 mV
    tau_rp = 2 * ms    # Refractory period set to 2 ms

    # Synapse parameters (adjusted for project: exc weight 0.5 mV, inh weight -2.5 mV)
    J = 0.5 * mV  # Excitatory synaptic weight
    g = 5         # Relative inhibitory strength (-g*J = -2.5 mV)
    D = 1.5 * ms  # Synaptic delay (unchanged from original)

    # External stimulus (adjusted for project: Poisson spikes at 1000 Hz per external connection)
    nu_ext = 1000 * Hz  # External input rate per connection

    defaultclock.dt = 0.1 * ms  # Time step set to 0.1 ms as specified

    # Neuron model adjusted to include resting potential E_L
    neurons = NeuronGroup(N,
                          """
                          dv/dt = (E_L - v)/tau : volt (unless refractory)
                          """,
                          threshold="v > theta",
                          reset="v = V_r",
                          refractory=tau_rp,
                          method="exact",
    )
    neurons.v = E_L  # Initial voltage set to resting potential

    excitatory_neurons = neurons[:N_E]
    inhibitory_neurons = neurons[N_E:]

    # Synaptic connections (unchanged structure, adjusted weights via J and g)
    exc_synapses = Synapses(excitatory_neurons, target=neurons, on_pre="v += J", delay=D)
    exc_synapses.connect(p=epsilon)

    inhib_synapses = Synapses(inhibitory_neurons, target=neurons, on_pre="v += -g*J", delay=D)
    inhib_synapses.connect(p=epsilon)

    # External Poisson input adjusted for specified rate
    external_poisson_input = PoissonInput(
        target=neurons, target_var="v", N=C_ext, rate=nu_ext, weight=J
    )

    # Monitors adjusted to record all neurons' spikes and voltages
    spike_monitor = SpikeMonitor(neurons)  # Records spikes from all 100 neurons
    state_monitor = StateMonitor(neurons, 'v', record=True, dt=1*ms)  # Records v at 1 ms resolution

    run(sim_time, report='text')

    # Save spike times (time in ms, neuron index) to output subfolder
    spike_times = np.column_stack((spike_monitor.t / ms, spike_monitor.i))
    np.savetxt(f'{output_dir}/run_{run_id}_spikes.dat', spike_times, fmt='%.3f %d')

    # Save voltage traces (time in ms, voltages in mV for all neurons) to output subfolder
    times = state_monitor.t / ms
    voltages = state_monitor.v / mV  # shape (n_neurons, n_times)
    voltage_data = np.column_stack((times, voltages.T))
    np.savetxt(f'{output_dir}/run_{run_id}_voltages.dat', voltage_data, fmt='%.3f')

# Run simulations for all run IDs after function definition
for run_id in run_ids:
    sim(sim_time, run_id)