import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import sax

from simphony.libraries import siepic
#sipann
from simphony.classical import ClassicalSim


from functools import partial


def ring_factory(wl=1.55, radius=10) -> sax.SDict:
    """Creates a full ring (with terminator) from a half ring.

    Resulting pins are ('through', 'drop', 'in', 'out').

    Parameters
    ----------
    wl : ArrayLike
        The wavelengths to simulate at, in microns.
    radius : float
        The radius of the ring resonator, in microns.
    """
    # Create rings for selecting out frequencies from the data line.
    # See simphony's SiPANN model library API for argument order and units.
    cir, info = sax.circuit(
        netlist={
            "instances": {
                "ring1": "half_ring",
                "ring2": "half_ring",
            },
            "connections": {
                "ring1,o1": "ring2,o3",
                "ring2,o1": "ring1,o3",
            },
            "ports": {
                "in": "ring1,o0",
                "through": "ring1,o2",
                "drop": "ring2,o2",
                "add": "ring2,o0",
            }
        },
        models={
            "half_ring": partial(sipann.half_ring, wl=wl, width=500, thickness=220, radius=radius, gap=100),
        }
    )
    # Return the composite model.
    return cir(wl=wl)

wl = np.linspace(1.5, 1.6, 1000)
ring1 = ring_factory(wl=wl, radius=10)

plt.plot(wl, np.abs(ring1['in', 'through'])**2, label="through")
plt.plot(wl, np.abs(ring1['in', 'drop'])**2, label="drop")
plt.plot(wl, np.abs(ring1['in', 'add'])**2, label="add")
plt.title("10-micron Ring Resonator")
plt.xlabel("Wavelength (microns)")
plt.legend()
plt.tight_layout()
#plt.show()
plt.savefig("plot.jpg")


