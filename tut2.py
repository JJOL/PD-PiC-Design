from jax import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import sax

from simphony.libraries import siepic

print(siepic.grating_coupler())

print("Grating Coupler Ports:")
print(sax.get_ports(siepic.grating_coupler))
print("Y-Branch Ports")
print(sax.get_ports(siepic.y_branch))


mzi, info = sax.circuit(
	netlist={
		"instances": {
			"gc_in": "gc",
			"splitter": "ybranch",
			"long_wg": "waveguide",
			"short_wg": "waveguide",
			"combiner": "ybranch",
			"gc_out": "gc",
		},
		"connections": {
			"gc_in,o0": "splitter,port 1",
			"splitter,port 2": "long_wg,o0",
			"splitter,port 3": "short_wg,o0",
			"long_wg,o1": "combiner,port 2",
			"short_wg,o1": "combiner,port 3",
			"combiner,port 1": "gc_out,o0",
		},
		"ports": {
			"in": "gc_in,o1",
			"out": "gc_out,o1",
		}
	},
	models={
		"ybranch": siepic.y_branch,
		"waveguide": siepic.waveguide,
		"gc": siepic.grating_coupler
	}
)


print("\n\nMZI Settings")
print(sax.get_settings(mzi))


wl = jnp.linspace(1.5, 1.6, 1000)
S = mzi(wl=wl, long_wg={"length": 150.0}, short_wg={"length": 50.0})

mag = jnp.abs(S["out","in"])**2
fig, axs = plt.subplots(2, 1, sharex=True)
axs[0].plot(wl, mag)
axs[0].set_ylabel("Transmission")

axs[1].plot(wl, 10*jnp.log10(mag))
axs[1].set_ylabel("Transmission (dB)")
axs[1].set_xlabel("Wavelength (um)")
plt.suptitle("MZI Response")
plt.savefig("mzi.jpg")



from simphony.classical import ClassicalSim
wl = jnp.linspace(1.5, 1.6, 1000)

sim = ClassicalSim(ckt=mzi, wl=wl, long_wg={"length": 150.0}, short_wg={"length": 50.0})
laser = sim.add_laser(ports=["in"], power=1.0)
detector = sim.add_detector(ports=["out"])
result = sim.run()

fig, axs = plt.subplots(1,1)
result.detectors["out"].plot(axs)
plt.title("MZI Sim Power")
plt.savefig("mzi_sim.jpg")


# Manually
wl = result.wl
s = result.sdict["out"]

plt.plot(wl, jnp.abs(s)**2)
plt.title("MZI s-parameter $|S{oi}|^2$")
plt.tight_layout()
plt.savefig("mzi_sim.jpg")
