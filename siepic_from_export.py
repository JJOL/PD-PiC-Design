from jax import config
config.update("jax_enable_x64", True)

import argparse
import ast
from collections import defaultdict

import jax.numpy as jnp
import matplotlib.pyplot as plt
import sax
from simphony.libraries import siepic

def greet():
    return "Hello from SiEPIC export PoC!"


class ExportedSubCircuit:

    def __init__(self, name, *external_nodes):
        self.name = name
        self.external_nodes = list(external_nodes)
        self.instances = []

    def X(self, instance_name, model_name, *nodes, **params):
        self.instances.append({
            "name": instance_name,
            "model": model_name,
            "nodes": list(nodes),
            "params": dict(params),
        })


class ExportedCircuit:

    def E(self, *args, **kwargs):
        # Ignored in this PoC: exported file may include one top-level call.
        return None


def _const_value(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
        return -node.operand.value
    raise ValueError(f"Unsupported AST node: {ast.dump(node)}")


def load_exported_subcircuit(path_or_content, is_file=True):
    if is_file:
        with open(path_or_content, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=path_or_content)
    else:
        tree = ast.parse(path_or_content, filename="<string>")

    env = {"SubCircuit": ExportedSubCircuit, "circuit": ExportedCircuit()}
    subcircuits = {}

    for stmt in tree.body:
        if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
            call = stmt.value
            if isinstance(call.func, ast.Name) and call.func.id == "SubCircuit":
                args = [_const_value(arg) for arg in call.args]
                obj = ExportedSubCircuit(*args)
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        env[target.id] = obj
                        subcircuits[target.id] = obj
            continue

        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            call = stmt.value
            if isinstance(call.func, ast.Attribute):
                receiver = call.func.value
                if isinstance(receiver, ast.Name) and receiver.id in env:
                    obj = env[receiver.id]
                    method_name = call.func.attr
                    args = [_const_value(arg) for arg in call.args]
                    kwargs = {kw.arg: _const_value(kw.value) for kw in call.keywords if kw.arg is not None}
                    getattr(obj, method_name)(*args, **kwargs)

    if not subcircuits:
        raise ValueError(f"No SubCircuit definition found in {path}")

    # PoC assumes a single subcircuit in exported file.
    return next(iter(subcircuits.values()))


def _model_key(model_name):
    model_name_l = model_name.lower()
    if model_name_l.startswith("gc_te_"):
        return "gc"
    if model_name_l == "ebeam_y_1550":
        return "ybranch"
    if model_name_l == "ebeam_wg_integral_1550":
        return "waveguide"
    raise ValueError(f"Unsupported model from export: {model_name}")


def _instance_ports(instance):
    model_key = _model_key(instance["model"])
    if model_key == "gc":
        return ["o1", "o0"]
    if model_key == "waveguide":
        return ["o0", "o1"]
    if model_key == "ybranch":
        return ["port 1", "port 2", "port 3"]
    raise ValueError(f"Unsupported model key: {model_key}")


def _friendly_port_name(raw_name):
    low = raw_name.lower()
    if "detector" in low:
        return "detector"
    if "laser" in low:
        return "laser"
    return raw_name


def build_sax_netlist(exported_subckt):
    instances = {}
    connections = {}
    ports = {}
    settings = {}

    node_to_endpoints = defaultdict(list)

    for instance in exported_subckt.instances:
        inst_name = instance["name"]
        model_key = _model_key(instance["model"])
        instances[inst_name] = model_key

        port_names = _instance_ports(instance)
        if len(port_names) != len(instance["nodes"]):
            raise ValueError(
                f"Port count mismatch for {inst_name}: model expects {len(port_names)}, got {len(instance['nodes'])}"
            )

        for node, port_name in zip(instance["nodes"], port_names):
            node_to_endpoints[node].append(f"{inst_name},{port_name}")

        if model_key == "waveguide":
            length_m = float(instance["params"].get("wg_length", 0.0))
            width_m = float(instance["params"].get("wg_width", 0.0))
            settings[inst_name] = {
                "length": length_m * 1e6,
                "width": width_m * 1e9,
            }

    for external_node in exported_subckt.external_nodes:
        endpoints = node_to_endpoints.get(external_node, [])
        if endpoints:
            ports[_friendly_port_name(external_node)] = endpoints[0]

    # Pairwise net connections for this PoC (MZI internal nets are pairwise).
    for node, endpoints in node_to_endpoints.items():
        if node in exported_subckt.external_nodes:
            continue
        if len(endpoints) == 2:
            connections[endpoints[0]] = endpoints[1]

    netlist = {
        "instances": instances,
        "connections": connections,
        "ports": ports,
    }

    return netlist, settings


def run_from_klayout(spice_filepath: str, output_plot_path: str):


    # FIRST WE HAVE TO RUN PySPICE 
    # python -m PySpice.Scripts.cir2py --allow-dollar-in-names --output loadable_netlist_path spice_filepath

    from PySpice.Spice.Parser import SpiceParser
    end_of_line_comment = ('//', ';')
    parser = SpiceParser(path=spice_filepath,
                         end_of_line_comment=end_of_line_comment)

    circuit_txt = parser.to_python_code(ground=0)

    exported_subckt = load_exported_subcircuit(circuit_txt, is_file=False)
    netlist, settings = build_sax_netlist(exported_subckt)

    models = {
        "gc": siepic.grating_coupler,
        "ybranch": siepic.y_branch,
        "waveguide": siepic.waveguide,
    }

    ckt, _ = sax.circuit(netlist=netlist, models=models)

    wl = jnp.linspace(1.5, 1.6, 1000)
    S = ckt(wl=wl, **settings)

    port_names = list(netlist["ports"].keys())
    if len(port_names) < 2:
        raise ValueError("Need at least two ports in exported netlist")
    in_port = "laser" if "laser" in netlist["ports"] else port_names[0]
    out_port = "detector" if "detector" in netlist["ports"] else port_names[1]
    mag = jnp.abs(S[out_port, in_port]) ** 2

    # set plotting backend to file non interactive environment
    plt.switch_backend('Agg')

    plt.plot(wl, mag)
    plt.xlabel("Wavelength (um)")
    plt.ylabel("Transmission")
    plt.title("SiEPIC-exported MZI Response")
    plt.tight_layout()
    plt.savefig(output_plot_path)

def main():
    parser = argparse.ArgumentParser(description="Run SiEPIC exported Python netlist through SAX")
    parser.add_argument(
        "--exported-netlist",
        default="mzi_sample1_spice.txt",
        help="Path to file containing exported SubCircuit/X calls",
    )
    parser.add_argument(
        "--output",
        default="siepic_mzi_response.jpg",
        help="Output plot path",
    )
    args = parser.parse_args()

    exported_subckt = load_exported_subcircuit(args.exported_netlist)
    netlist, settings = build_sax_netlist(exported_subckt)

    models = {
        "gc": siepic.grating_coupler,
        "ybranch": siepic.y_branch,
        "waveguide": siepic.waveguide,
    }

    ckt, _ = sax.circuit(netlist=netlist, models=models)

    wl = jnp.linspace(1.5, 1.6, 1000)
    S = ckt(wl=wl, **settings)

    port_names = list(netlist["ports"].keys())
    if len(port_names) < 2:
        raise ValueError("Need at least two ports in exported netlist")
    in_port = "laser" if "laser" in netlist["ports"] else port_names[0]
    out_port = "detector" if "detector" in netlist["ports"] else port_names[1]
    mag = jnp.abs(S[out_port, in_port]) ** 2

    plt.plot(wl, mag)
    plt.xlabel("Wavelength (um)")
    plt.ylabel("Transmission")
    plt.title("SiEPIC-exported MZI Response")
    plt.tight_layout()
    plt.savefig(args.output)


if __name__ == "__main__":
    main()
