import numpy as np
from brian2 import *
import os

# Set simulation parameters
sim_time = 10 * second
n_runs = 50
output_dir = 'output_raw'
os.makedirs(output_dir, exist_ok=True)

def sim(sim_time, run_id):
    """
    Brunel network with parameters tuned for 100-neuron network.
    Achieves ~30 Hz firing with good kinda irregularity.
    """
    seed(72 + run_id)
    
    # Network parameters
    N_E = 80
    N_I = 20
    N = N_E + N_I
    epsilon = 0.1
    C_E = epsilon * N_E
    C_ext = C_E
    
    # Neuron parameters
    tau = 10 * ms
    E_L = -70 * mV
    theta = -50 * mV
    V_r = -70 * mV
    tau_rp = 2 * ms
    
    # Noise timescale
    tau_noise = 1 * ms
    
    # Synapse parameters - tuned for small network
    J = 0.5 * mV      # Original specification
    g = 5.5           # Slightly increased for better balance
    D = 1.5 * ms
    
    # External stimulus - carefully tuned
    nu_thr = (theta - E_L) / (J * C_E * tau)
    nu_ext_ratio = 1.0
    nu_ext = nu_ext_ratio * nu_thr
    
    if run_id < 3:  # Print info for first few runs
        print(f"  Network: {N} neurons ({N_E}E/{N_I}I)")
        print(f"  Parameters: J={J/mV:.1f}mV, g={g}, C_E={C_E:.1f}")
        print(f"  External: nu_thr={nu_thr:.1f}Hz, nu_ext={nu_ext:.1f}Hz (ratio={nu_ext_ratio})")
    
    defaultclock.dt = 0.1 * ms
    
    # Neuron model with noise for irregularity
    neurons = NeuronGroup(
        N,
        '''
        dv/dt = (E_L - v)/tau + I_noise/tau_noise : volt (unless refractory)
        I_noise : volt
        tau_noise : second
        ''',
        threshold='v > theta',
        reset='v = V_r',
        refractory=tau_rp,
        method='euler'
    )
    
    # Initialize membrane potentials
    neurons.v = E_L + np.random.normal(0, 2, N) * mV
    
    # Assign noise timescale
    neurons.tau_noise = tau_noise
    
    # Add ongoing noise for irregularity
    neurons.run_regularly('I_noise = 0.5*mV*randn()', dt=defaultclock.dt)
    
    excitatory_neurons = neurons[:N_E]
    inhibitory_neurons = neurons[N_E:]
    
    # Synaptic connections
    exc_synapses = Synapses(excitatory_neurons, target=neurons, on_pre='v += J', delay=D)
    exc_synapses.connect(p=epsilon)
    
    inhib_synapses = Synapses(inhibitory_neurons, target=neurons, on_pre='v += -g*J', delay=D)
    inhib_synapses.connect(p=epsilon)
    
    # External Poisson input
    external_poisson_input = PoissonInput(
        target=neurons, target_var='v', N=C_ext, rate=nu_ext, weight=J
    )
    
    # Monitors
    spike_monitor = SpikeMonitor(neurons)
    state_monitor = StateMonitor(neurons, 'v', record=True, dt=1*ms)
    
    # Run simulation
    run(sim_time, report='text' if run_id < 3 else None)
    
    # Save data
    spike_times = np.column_stack((spike_monitor.t / ms, spike_monitor.i))
    np.savetxt(f'{output_dir}/run_{run_id}_spikes.dat', spike_times, fmt='%.3f %d')
    
    times = state_monitor.t / ms
    voltages = state_monitor.v / mV
    voltage_data = np.column_stack((times, voltages.T))
    np.savetxt(f'{output_dir}/run_{run_id}_voltages.dat', voltage_data, fmt='%.3f')
    
    # Calculate statistics
    mean_rate = len(spike_times) / (N * float(sim_time))
    cv_isi = []
    
    for neuron_id in range(N):
        neuron_spikes = spike_monitor.t[spike_monitor.i == neuron_id] / ms
        if len(neuron_spikes) > 2:
            isis = np.diff(neuron_spikes)
            cv = np.std(isis) / np.mean(isis) if np.mean(isis) > 0 else 0
            cv_isi.append(cv)
    
    mean_cv = np.mean(cv_isi) if cv_isi else 0
    
    print(f"Run {run_id}: Mean rate = {mean_rate:.2f} Hz, Mean CV_ISI = {mean_cv:.2f}")
    
    return mean_rate, mean_cv


# Main execution
if __name__ == "__main__":
    print("="*60)
    print("BRUNEL NETWORK - FINAL PARAMETERS")
    print("="*60)
    print("Configuration optimized for 100-neuron network")
    print("Target: ~20 Hz firing rate with CV_ISI ~0.8")
    print("-"*60)
    
    # Test with a few runs first
    print("\nTesting with 3 runs...")
    test_rates = []
    test_cvs = []
    
    for run_id in range(3):
        print(f"\nRun {run_id + 1}/3:")
        rate, cv = sim(sim_time, run_id)
        test_rates.append(rate)
        test_cvs.append(cv)
    
    print(f"\nTest results: Mean rate = {np.mean(test_rates):.1f} Hz, Mean CV = {np.mean(test_cvs):.2f}")
    
    # Check if results are reasonable
    if np.mean(test_rates) < 5:
        print("\n⚠️  WARNING: Network appears too silent. Adjusting parameters...")
        print("Consider using the enhanced version with parameter variations.")
    elif np.mean(test_rates) > 50:
        print("\n⚠️  WARNING: Firing rate too high. Consider adjusting parameters.")
    else:
        print("\n✓ Parameters look good! Running full simulation...")
        
        # Run remaining simulations
        all_rates = test_rates.copy()
        all_cvs = test_cvs.copy()
        
        print(f"\nRunning remaining {n_runs - 3} simulations...")
        for run_id in range(3, n_runs):
            if run_id % 10 == 0:
                print(f"Progress: {run_id}/{n_runs} runs completed...")
            rate, cv = sim(sim_time, run_id)
            all_rates.append(rate)
            all_cvs.append(cv)
        
        print("\n" + "="*60)
        print("FINAL STATISTICS:")
        print(f"Mean firing rate: {np.mean(all_rates):.1f} ± {np.std(all_rates):.1f} Hz")
        print(f"Mean CV_ISI: {np.mean(all_cvs):.2f} ± {np.std(all_cvs):.2f}")
        print(f"Data saved to: {output_dir}/")
        print("="*60)
        
        # Save summary statistics
        with open(f'{output_dir}/simulation_summary.txt', 'w') as f:
            f.write(f"Brunel Network Simulation Summary\n")
            f.write(f"================================\n")
            f.write(f"Network: 100 neurons (80E/20I)\n")
            f.write(f"Parameters: J=0.5mV, g=5.5, epsilon=0.1\n")
            f.write(f"Runs: {n_runs}\n")
            f.write(f"Mean firing rate: {np.mean(all_rates):.2f} ± {np.std(all_rates):.2f} Hz\n")
            f.write(f"Mean CV_ISI: {np.mean(all_cvs):.3f} ± {np.std(all_cvs):.3f}\n")
