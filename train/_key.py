from nequip.data import AtomicDataDict

RMSE_LOSS_KEY = "rmse"
MAE_KEY = "mae"
LOSS_KEY = "noramlized_loss"

VALUE_KEY = "value"
CONTRIB = "contrib"

VALIDATION = "validation"
TRAIN = "training"

ABBREV = {
    AtomicDataDict.TOTAL_ENERGY_KEY: "e",
    AtomicDataDict.TOTAL_ENERGY_FIX_SPIN_KEY: "e_fix_s",
    AtomicDataDict.TOTAL_ENERGY_DELTA_SPIN_KEY: "e_del_s",
    AtomicDataDict.PER_ATOM_ENERGY_KEY: "Ei",
    AtomicDataDict.FORCE_KEY: "f",
    AtomicDataDict.NODE_FEATURES_KEY: "h",
    AtomicDataDict.SPIN_FORCE_KEY: "spin_f",
    AtomicDataDict.SPIN_FORCE_VERT_KEY: "spin_fv",
    AtomicDataDict.BORN_EFFECTIVE_CHARGE_KEY: "bec",
    AtomicDataDict.TOTAL_BORN_EFFECTIVE_CHARGE_KEY: "bec_0",
    AtomicDataDict.ATOMIC_DIPOLE_KEY: "atom_dip",
    AtomicDataDict.TOTAL_DIPOLE_KEY: "tot_dip",
    AtomicDataDict.PIEZOELECTRIC_KEY: "piezo",
    LOSS_KEY: "loss",
    VALIDATION: "val",
    TRAIN: "train",
}
