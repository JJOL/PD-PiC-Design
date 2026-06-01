import numpy as np
from jax import config
import sax
config.update("jax_enable_x64", True)

print("Hello, World!")

s = {
	("in", "in"): np.array([[0.5]]),
    ("in", "out"): np.array([[0.5]]),
	("out", "in"): np.array([[0.5]]),
	("out", "out"): np.array([[0.5]]),
}

print("Port in-out:", s[("in","out")])


def custom_model(param: float = 0.5) -> sax.SDict:
	sdict = sax.reciprocal({
		("in0","out0"): -1j * np.sqrt(param),
	})
	return sdict

print("custom_model(param=0.25):", custom_model(param=0.25))

from simphony.libraries import siepic

wg1 = siepic.waveguide(length=2500, height=220)
wg2 = siepic.waveguide(length=7500, height=210)

print("wg1:", wg1)   #o0<->o1 S: -0.82 + 0.5j
print("wg2:", wg2)   #o0<->o1 S: 0.99 + 0.10j

netlist = {
	"instances": {
		"wg1": "waveguide",
		"wg2": "waveguide",
	},
	"connections": {
		"wg1,o1": "wg2,o0",
	},
	"ports": {
		"in": "wg1,o0",
		"out": "wg2,o1",
	}
}

circuit, info = sax.circuit(
	netlist=netlist,
	models={
		"waveguide": siepic.waveguide,
	}
)


print("\n\nCircuit Info:", info)
print("Circuit:", circuit)


print("\n\nParameter Evaluation")
wls = np.linspace(1.5, 1.6, 5)
print("WLs: ", wls)
sdict = circuit(wl=wls)
print(sdict)



from simphony.classical import ClassicalSim

sim = ClassicalSim(circuit, wl=1.55,
	wg1={"length": 2500.0, "loss": 3.0}, wg2={"length": 7500.0, "loss": 3.0}
)
laser = sim.add_laser(ports=["in"], power=1.0)
detector = sim.add_detector(ports=["out"])

result = sim.run()

print(f"Power transmission: {abs(result.sdict['out'])**2}")





