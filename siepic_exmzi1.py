from jax import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import sax
from simphony.libraries import siepic

netlist = {
	"instances": {
		"gc_detector": "gc",
		"gc_laser": "gc",
		"splitter": "ybranch",
		"combiner": "ybranch",
		"wg_detector_to_splitter": "waveguide",
		"wg_laser_to_combiner": "waveguide",
		"wg_upper_arm": "waveguide",
		"wg_lower_arm": "waveguide",
	},
	"connections": {
		"gc_detector,o0": "wg_detector_to_splitter,o0",
		"wg_detector_to_splitter,o1": "splitter,port 1",
		
		"gc_laser,o0": "wg_laser_to_combiner,o0",
		"wg_laser_to_combiner,o1": "combiner,port 1",
		
		"splitter,port 2": "wg_upper_arm,o0",
		"wg_upper_arm,o1": "combiner,port 3",

		"splitter,port 3": "wg_lower_arm,o0",
		"wg_lower_arm,o1": "combiner,port 2",
	},
	"ports": {
		"detector": "gc_detector,o1",
		"laser": "gc_laser,o1",
	}
}
models = {
	"gc": siepic.grating_coupler,
	"ybranch": siepic.y_branch,
	"waveguide": siepic.waveguide,
}


ckt, info = sax.circuit(netlist=netlist, models=models)

settings = {
	"wg_detector_to_splitter": {
		"length": 12.600,
		"width": 500
	},
	"wg_laser_to_combiner": {
		"length": 26.897,
		"width": 500,
	},
	"wg_upper_arm": {
		"length": 125.797,
		"width": 500,
	},
	"wg_lower_arm": {
		"length": 176.797,
		"width": 500,
	}
}

wl = jnp.linspace(1.5, 1.6, 1000)
S = ckt(wl=wl,
	wg_detector_to_splitter=settings["wg_detector_to_splitter"],
    wg_laser_to_combiner=settings["wg_laser_to_combiner"],
    wg_upper_arm=settings["wg_upper_arm"],
    wg_lower_arm=settings["wg_lower_arm"],
)

mag = jnp.abs(S["detector", "laser"])**2

plt.plot(wl, mag)
plt.xlabel("Wavelength (um)")
plt.ylabel("Transmission")
plt.title("SiEPIC-exported Example MZI 1 Response")
plt.tight_layout()
plt.savefig("siepic_mzi_response.jpg")
