"""Generate a synthetic cantilever-beam deflection dataset for the example.

This is fully synthetic (no proprietary or third-party data): it samples random
beam geometries and loads, computes the analytic tip deflection of a cantilever
beam under a point load,

    delta = P * L**3 / (3 * E * I)

and adds Gaussian noise to represent the sort of scatter you would see in a real
simulation or physical experiment. Run this script to (re)create
``examples/beam_deflection/data.csv``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_STATE = 42
N_SAMPLES = 600

# Elastic modulus (Pa) by material, representative order-of-magnitude values.
MATERIAL_MODULUS_PA = {
    "steel": 200e9,
    "aluminum": 69e9,
    "composite": 45e9,
}


def generate(n_samples: int = N_SAMPLES, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    length_m = rng.uniform(1.0, 2.0, n_samples)
    load_n = rng.uniform(200.0, 800.0, n_samples)
    moment_of_inertia_m4 = rng.uniform(8e-7, 2e-6, n_samples)
    material = rng.choice(list(MATERIAL_MODULUS_PA), size=n_samples)
    modulus_pa = np.array([MATERIAL_MODULUS_PA[m] for m in material])

    deflection_m = (load_n * length_m**3) / (3 * modulus_pa * moment_of_inertia_m4)
    # Multiplicative (relative) noise, representative of measurement/simulation noise that
    # scales with the signal, rather than fixed-magnitude noise that would swamp small
    # deflections and be negligible for large ones (and could make deflection go negative,
    # which is not physical for this quantity).
    noise_factor = 1.0 + rng.normal(loc=0.0, scale=0.05, size=n_samples)
    deflection_m_noisy = deflection_m * noise_factor

    return pd.DataFrame(
        {
            "length_m": length_m,
            "load_n": load_n,
            "moment_of_inertia_m4": moment_of_inertia_m4,
            "material": material,
            "deflection_m": deflection_m_noisy,
        }
    )


if __name__ == "__main__":
    df = generate()
    output_path = Path(__file__).parent / "data.csv"
    df.to_csv(output_path, index=False)
    print(f"Wrote {len(df)} rows to {output_path}")
