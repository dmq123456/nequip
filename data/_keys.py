"""Keys for dictionaries/AtomicData objects.

This is a seperate module to compensate for a TorchScript bug that can only recognize constants when they are accessed as attributes of an imported module.
"""
import sys
from typing import List

if sys.version_info[1] >= 8:
    from typing import Final
else:
    from typing_extensions import Final

# == Define allowed keys as constants ==
# The positions of the atoms in the system
POSITIONS_KEY: Final[str] = "pos"
# The [2, n_edge] index tensor giving center -> neighbor relations
EDGE_INDEX_KEY: Final[str] = "edge_index"
# A [n_edge, 3] tensor of how many periodic cells each edge crosses in each cell vector
EDGE_CELL_SHIFT_KEY: Final[str] = "edge_cell_shift"
# [n_batch, 3, 3] or [3, 3] tensor where rows are the cell vectors
CELL_KEY: Final[str] = "cell"
# [n_batch, 3] bool tensor
PBC_KEY: Final[str] = "pbc"
# [n_atom, 1] long tensor
ATOMIC_NUMBERS_KEY: Final[str] = "atomic_numbers"
# [n_atom, 1] long tensor
ATOM_TYPE_KEY: Final[str] = "atom_types"
# The spin of the atoms in the system
SPIN_KEY: Final[str] = "spin"  # original spin input as leaf variable and can autograd
SPIN_LENGTH_KEY: Final[str] = "spin_length" # length of spin, magnitude of spin
SPIN_NONLEAF_KEY: Final[
    str
] = "spin_nonleaf"  # processed spin. Not a leaf variable. Can not autograd

BASIC_STRUCTURE_KEYS: Final[List[str]] = [
    POSITIONS_KEY,
    EDGE_INDEX_KEY,
    EDGE_CELL_SHIFT_KEY,
    CELL_KEY,
    PBC_KEY,
    ATOM_TYPE_KEY,
    ATOMIC_NUMBERS_KEY,
    SPIN_KEY,
    SPIN_LENGTH_KEY,
    SPIN_NONLEAF_KEY,
]

# A [n_edge, 3] tensor of displacement vectors associated to edges
EDGE_VECTORS_KEY: Final[str] = "edge_vectors"
# A [n_edge] tensor of the lengths of EDGE_VECTORS
EDGE_LENGTH_KEY: Final[str] = "edge_lengths"
# [n_edge, dim] (possibly equivariant) attributes of each edge
EDGE_ATTRS_KEY: Final[str] = "edge_attrs"
# [n_edge, dim] invariant embedding of the edges
EDGE_EMBEDDING_KEY: Final[str] = "edge_embedding"
EDGE_FEATURES_KEY: Final[str] = "edge_features"
EDGE_NONSCALAR_FEATURES_KEY: Final[str] = "edge_nonscalar_features"

NODE_FEATURES_KEY: Final[str] = "node_features"
NODE_ATTRS_KEY: Final[str] = "node_attrs"

PER_ATOM_ENERGY_KEY: Final[str] = "atomic_energy"
TOTAL_ENERGY_KEY: Final[str] = "total_energy"
TOTAL_ENERGY_FIX_SPIN_KEY: Final[str] = "energy_fix_spin"
TOTAL_ENERGY_DELTA_SPIN_KEY: Final[str] = "energy_delta_spin"
FORCE_KEY: Final[str] = "forces"
PARTIAL_FORCE_KEY: Final[str] = "partial_forces"
SPIN_FORCE_KEY: Final[str] = "spin_forces"
SPIN_FORCE_VERT_KEY: Final[str] = "spin_forces_vert"
SPIN_PARTIAL_FORCE_KEY: Final[str] = "spin_partial_forces"
STRESS_KEY: Final[str] = "stress"
VIRIAL_KEY: Final[str] = "virial"
ATOM_VIRIAL_KEY: Final[str] = "atom_virial"
BORN_EFFECTIVE_CHARGE_KEY: Final[str] = "born_effective_charges" # In units of e
TOTAL_BORN_EFFECTIVE_CHARGE_KEY: Final[str] = "total_born_effective_charges" # In units of e
PIEZOELECTRIC_KEY: Final[str] = "piezoelectric" # In units of e*Angstrom, (3, 3, 3), first index for E-field, second and third for related virial.

ATOMIC_DIPOLE_KEY: Final[str] = "atomic_dipole" # In units of e*Angstrom
TOTAL_DIPOLE_KEY: Final[str] = "dipole" # In units of e*Angstrom

EDGE_ENERGY_KEY: Final[str] = "edge_energy"
EDGE_HB_COEFF_KEY: Final[str] = "edge_hb_coeff"
EDGE_BIQUAD_COEFF_KEY: Final[str] = "edge_biquad_coeff"
EDGE_SPIN_MATRIX_KEY: Final[str] = "edge_spin_matrix"
NODE_SPINANI_MATRIX_KEY: Final[str] = "node_spinani_matrix"
EDGE_HB_MATRIX_KEY: Final[str] = "edge_hb_matrix"
EDGE_SISJ_KEY: Final[str] = "edge_sisj"

ALL_ENERGY_KEYS: Final[List[str]] = [
    PER_ATOM_ENERGY_KEY,
    TOTAL_ENERGY_KEY,
    TOTAL_ENERGY_FIX_SPIN_KEY,
    TOTAL_ENERGY_DELTA_SPIN_KEY,
    FORCE_KEY,
    PARTIAL_FORCE_KEY,
    SPIN_FORCE_KEY,
    SPIN_FORCE_VERT_KEY,
    SPIN_PARTIAL_FORCE_KEY,
    STRESS_KEY,
    VIRIAL_KEY,
    ATOM_VIRIAL_KEY,
    EDGE_ENERGY_KEY,
    EDGE_HB_COEFF_KEY,
    EDGE_BIQUAD_COEFF_KEY,
    # EDGE_SPIN_MATRIX_KEY,
    EDGE_HB_MATRIX_KEY,
    NODE_SPINANI_MATRIX_KEY,
]

BATCH_KEY: Final[str] = "batch"

# Make a list of allowed keys
ALLOWED_KEYS: List[str] = [
    getattr(sys.modules[__name__], k)
    for k in sys.modules[__name__].__dict__.keys()
    if k.endswith("_KEY")
]
