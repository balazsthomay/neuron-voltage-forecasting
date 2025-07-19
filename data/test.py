from brian2 import *

# A simple leaky integrate-and-fire model
eqs = '''
dv/dt = (1-v)/tau : 1
tau : second
'''

G = NeuronGroup(1, eqs, threshold='v>0.8', reset='v=0', method='exact')
G.tau = 10*ms

M = StateMonitor(G, 'v', record=True)

print("Running simulation...")
run(100*ms)
print("Simulation finished.")

# Plot the results
plot(M.t/ms, M.v[0])
xlabel('Time (ms)')
ylabel('v')
show()

print("Plotting complete. Everything seems to work!")