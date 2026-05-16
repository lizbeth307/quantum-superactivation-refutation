from .math import von_neumann_entropy, batch_kron, make_tp_kraus, evaluate_stinespring_capacity, evaluate_complementary_capacity, evaluate_npt_penalty, custom_eigh
from .channels import build_erasure_channel, build_depolarizing_channel, build_smith_yard_ppt, build_phase_damping_channel, build_amplitude_damping_channel, build_black_hole_channel, build_wormhole_channel
from .io import QuantumStorage

__version__ = "2.0.0"
__author__ = "Quantum Superactivation Engine"
