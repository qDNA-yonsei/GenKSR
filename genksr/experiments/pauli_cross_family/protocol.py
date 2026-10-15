"""Frozen protocol construction for the Pauli cross-family experiment.

The experiment deliberately uses a family-agnostic Hamiltonian interface.  A
Hamiltonian is represented only by its unordered set of Pauli strings and real
coefficients; no family identifier is exposed to the model.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import yaml
from scipy.stats import qmc


PROTOCOL_VERSION = 4
SUPPORTED_PROTOCOL_VERSIONS = (1, 2, 3, 4)
SOURCE_FAMILIES = ("tfim", "xxz", "xy_staggered_z")
TARGET_FAMILY = "mixed_field_xxz"
PAULI_TO_CODE = {"I": 0, "X": 1, "Y": 2, "Z": 3}
CODE_TO_PAULI = tuple(PAULI_TO_CODE)

FAMILY_SPECS = {
    "tfim": {
        "parameters": ("zz", "x"),
        "expression": "a B_ZZ + b B_X",
        "role": "source",
    },
    "xxz": {
        "parameters": ("xy", "zz"),
        "expression": "a B_XY + b B_ZZ",
        "role": "source",
    },
    "xy_staggered_z": {
        "parameters": ("xy", "zs"),
        "expression": "a B_XY + b B_Zs",
        "role": "source",
    },
    "mixed_field_xxz": {
        "parameters": ("xy", "zz", "x", "zs"),
        "expression": "a B_XY + b B_ZZ + c B_X + d B_Zs",
        "role": "target",
    },
    "staggered_field_xxz": {
        "parameters": ("xy", "zz", "zs"),
        "expression": "a B_XY + b B_ZZ + c B_Zs",
        "role": "target",
    },
    "transverse_field_xxz": {
        "parameters": ("xy", "zz", "x"),
        "expression": "a B_XY + b B_ZZ + c B_X",
        "role": "target",
    },
}


def _pauli_string(n_qubit: int, assignments: Mapping[int, str]) -> str:
    symbols = ["I"] * n_qubit
    for site, symbol in assignments.items():
        if not 0 <= int(site) < n_qubit:
            raise ValueError(f"site {site} outside n={n_qubit}")
        if symbol not in PAULI_TO_CODE:
            raise ValueError(f"unknown Pauli symbol {symbol!r}")
        symbols[int(site)] = symbol
    return "".join(symbols)


def canonicalize_terms(
    terms: Iterable[tuple[str, float]], *, atol: float = 1e-14
) -> list[tuple[str, float]]:
    """Merge duplicate strings, remove identity/zero terms, and sort.

    String position ``i`` is logical qubit ``i`` and is also bit position ``i``
    in displayed measurement arrays.  Conversion to integer basis labels treats
    position zero as the most-significant bit.
    """
    combined: dict[str, float] = defaultdict(float)
    n_qubit = None
    for pauli, coefficient in terms:
        pauli = str(pauli)
        if n_qubit is None:
            n_qubit = len(pauli)
        if len(pauli) != n_qubit or not pauli:
            raise ValueError("all Pauli strings must have one common positive length")
        if any(symbol not in PAULI_TO_CODE for symbol in pauli):
            raise ValueError(f"invalid Pauli string {pauli!r}")
        coefficient = float(coefficient)
        if not np.isfinite(coefficient):
            raise ValueError("Pauli coefficients must be finite")
        combined[pauli] += coefficient

    if n_qubit is None:
        raise ValueError("Hamiltonian must contain at least one candidate term")
    identity = "I" * n_qubit
    output = [
        (pauli, float(coefficient))
        for pauli, coefficient in combined.items()
        if pauli != identity and not np.isclose(coefficient, 0.0, atol=atol, rtol=0.0)
    ]
    if not output:
        raise ValueError("Hamiltonian is empty after canonicalization")
    return sorted(output, key=lambda item: item[0])


def _block_xy(n_qubit: int, coefficient: float) -> list[tuple[str, float]]:
    terms = []
    for site in range(n_qubit - 1):
        terms.append((_pauli_string(n_qubit, {site: "X", site + 1: "X"}), coefficient))
        terms.append((_pauli_string(n_qubit, {site: "Y", site + 1: "Y"}), coefficient))
    return terms


def _block_zz(n_qubit: int, coefficient: float) -> list[tuple[str, float]]:
    return [
        (_pauli_string(n_qubit, {site: "Z", site + 1: "Z"}), coefficient)
        for site in range(n_qubit - 1)
    ]


def _block_x(n_qubit: int, coefficient: float) -> list[tuple[str, float]]:
    return [
        (_pauli_string(n_qubit, {site: "X"}), coefficient)
        for site in range(n_qubit)
    ]


def _block_staggered_z(n_qubit: int, coefficient: float) -> list[tuple[str, float]]:
    return [
        # The displayed-order reference is 0101..., whose Z eigenvalues are
        # (+1,-1,+1,-1,...).  The minus sign makes a positive coefficient favor
        # that fixed reference instead of its orthogonal translated partner.
        (_pauli_string(n_qubit, {site: "Z"}), -coefficient * ((-1.0) ** site))
        for site in range(n_qubit)
    ]


def build_family_terms(
    family: str, n_qubit: int, parameters: Mapping[str, float]
) -> list[tuple[str, float]]:
    """Construct one open-chain family instance as canonical Pauli terms."""
    if family not in FAMILY_SPECS:
        raise ValueError(f"unknown family {family!r}")
    if n_qubit < 2:
        raise ValueError("cross-family experiment requires at least two qubits")
    expected = set(FAMILY_SPECS[family]["parameters"])
    received = set(parameters)
    if received != expected:
        raise ValueError(
            f"{family} parameters must be {sorted(expected)}, got {sorted(received)}"
        )
    values = {name: float(parameters[name]) for name in expected}
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError("family parameters must be finite")

    terms: list[tuple[str, float]] = []
    if "xy" in values:
        terms.extend(_block_xy(n_qubit, values["xy"]))
    if "zz" in values:
        terms.extend(_block_zz(n_qubit, values["zz"]))
    if "x" in values:
        terms.extend(_block_x(n_qubit, values["x"]))
    if "zs" in values:
        terms.extend(_block_staggered_z(n_qubit, values["zs"]))
    return canonicalize_terms(terms)


def coefficient_l1(terms: Sequence[tuple[str, float]]) -> float:
    value = float(sum(abs(float(coefficient)) for _, coefficient in terms))
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("Pauli coefficient L1 norm must be finite and positive")
    return value


def encode_terms(
    terms: Sequence[tuple[str, float]], *, normalize: bool = True
) -> tuple[np.ndarray, np.ndarray, float]:
    canonical = canonicalize_terms(terms)
    scale = coefficient_l1(canonical)
    codes = np.asarray(
        [[PAULI_TO_CODE[symbol] for symbol in pauli] for pauli, _ in canonical],
        dtype=np.uint8,
    )
    coefficients = np.asarray([value for _, value in canonical], dtype=np.float64)
    if normalize:
        coefficients = coefficients / scale
    return codes, coefficients, scale


def family_support(family: str, n_qubit: int) -> frozenset[str]:
    parameters = {name: 1.0 for name in FAMILY_SPECS[family]["parameters"]}
    return frozenset(pauli for pauli, _ in build_family_terms(family, n_qubit, parameters))


def _stable_seed(master_seed: int, *parts: object) -> int:
    payload = "|".join([str(int(master_seed)), *(str(part) for part in parts)]).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32)


def _sobol_parameters(
    names: Sequence[str], count: int, coefficient_range: Sequence[float], seed: int
) -> list[dict[str, float]]:
    if count <= 0:
        raise ValueError("Sobol count must be positive")
    low, high = map(float, coefficient_range)
    if not np.isfinite(low) or not np.isfinite(high) or not low < high:
        raise ValueError("coefficient_range must be finite and increasing")
    sampler = qmc.Sobol(d=len(names), scramble=True, seed=int(seed))
    # Draw the next complete base-2 Sobol net and truncate it.  This preserves
    # better space-filling behavior than sampler.random(n=count) for a
    # non-power-of-two count such as the 192-Hamiltonian matched-budget oracle.
    exponent = int(math.ceil(math.log2(count)))
    points = sampler.random_base2(m=exponent)[:count]
    points = qmc.scale(points, low, high)
    return [
        {name: float(value) for name, value in zip(names, point)}
        for point in points
    ]


def load_experiment_config(path: str | Path, *, smoke: bool = False) -> dict:
    path = Path(path)
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise ValueError("experiment config must be a mapping")
    config = deepcopy(config)
    if smoke:
        overrides = config.get("smoke")
        if not isinstance(overrides, dict):
            raise ValueError("smoke config is missing")
        for section, values in overrides.items():
            if section not in config or not isinstance(config[section], dict):
                raise ValueError(f"cannot apply smoke section {section!r}")
            config[section].update(values)
        config["experiment_name"] = f"{config['experiment_name']}_smoke"
    config.pop("smoke", None)
    return config


def _validated_config(config: Mapping) -> dict:
    config = deepcopy(dict(config))
    required_sections = ("data", "model", "training", "evaluation")
    for section in required_sections:
        if not isinstance(config.get(section), dict):
            raise ValueError(f"missing config section {section!r}")

    data = config["data"]
    required_data = (
        "n_qubit",
        "depth",
        "train_shots_per_condition",
        "source_train_per_family",
        "source_holdout_per_family",
        "target_test_hamiltonians",
        "coefficient_range",
        "master_seed",
    )
    for name in required_data:
        if name not in data:
            raise ValueError(f"missing data.{name}")
    for name in (
        "n_qubit",
        "depth",
        "train_shots_per_condition",
        "source_train_per_family",
        "source_holdout_per_family",
        "target_test_hamiltonians",
    ):
        data[name] = int(data[name])
        if data[name] <= 0:
            raise ValueError(f"data.{name} must be positive")
    if data["n_qubit"] % 2:
        raise ValueError("the fixed Neel reference requires an even number of qubits")
    data["master_seed"] = int(data["master_seed"])
    if len(data["coefficient_range"]) != 2:
        raise ValueError("data.coefficient_range must contain [low, high]")
    return config


def build_protocol(config: Mapping) -> dict:
    """Build a deterministic, serializable train/holdout manifest."""
    config = _validated_config(config)
    data = config["data"]
    n_qubit = data["n_qubit"]
    master_seed = data["master_seed"]
    coefficient_range = data["coefficient_range"]
    source_families = tuple(config.get("source_families", SOURCE_FAMILIES))
    target_family = str(config.get("target_family", TARGET_FAMILY))
    allow_target_template_in_source = bool(
        config.get("allow_target_template_in_source", False)
    )
    if not source_families or len(source_families) != len(set(source_families)):
        raise ValueError("source_families must be a non-empty unique sequence")
    if any(family not in FAMILY_SPECS for family in source_families):
        raise ValueError("source_families contains an unknown family")
    if target_family not in FAMILY_SPECS:
        raise ValueError(f"unknown target_family {target_family!r}")
    if target_family in source_families and not allow_target_template_in_source:
        raise ValueError("target_family must be absent from source_families")
    if allow_target_template_in_source and target_family not in source_families:
        raise ValueError(
            "allow_target_template_in_source requires target_family to be a "
            "source family"
        )

    split_specs = []
    for family in source_families:
        split_specs.extend(
            [
                ("source_train", family, data["source_train_per_family"]),
                ("source_holdout", family, data["source_holdout_per_family"]),
            ]
        )
    split_specs.append(("target_test", target_family, data["target_test_hamiltonians"]))

    instances = []
    for split, family, count in split_specs:
        parameter_names = FAMILY_SPECS[family]["parameters"]
        parameter_sets = _sobol_parameters(
            parameter_names,
            count,
            coefficient_range,
            _stable_seed(master_seed, "sobol", split, family),
        )
        for index, parameters in enumerate(parameter_sets):
            terms = build_family_terms(family, n_qubit, parameters)
            scale = coefficient_l1(terms)
            instances.append(
                {
                    "instance_id": f"{split}__{family}__{index:04d}",
                    "split": split,
                    "family": family,
                    "index": index,
                    "parameters": parameters,
                    "num_terms": len(terms),
                    "coefficient_l1": scale,
                    "dt": math.pi / scale,
                    "instance_seed": _stable_seed(master_seed, "instance", split, family, index),
                }
            )

    source_support_union = frozenset().union(
        *(family_support(family, n_qubit) for family in source_families)
    )
    target_support = family_support(target_family, n_qubit)
    if not target_support <= source_support_union:
        missing = sorted(target_support - source_support_union)
        raise AssertionError(f"target primitive strings absent from source support: {missing}")
    source_templates = {family_support(family, n_qubit) for family in source_families}
    target_template_seen = target_support in source_templates
    if target_template_seen and not allow_target_template_in_source:
        raise AssertionError("target operator template is present among source templates")

    evaluation_regime = (
        "in_family_target_trained_control"
        if target_template_seen
        else "zero_shot_operator_template_holdout"
    )

    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "experiment_name": str(config.get("experiment_name", "pauli_cross_family")),
        "description": (
            f"Budget-matched in-family target-trained control for {target_family}."
            if target_template_seen
            else f"Zero-shot compositional transfer from {len(source_families)} "
            f"source Pauli operator templates to held-out {target_family} at fixed size."
        ),
        "config": config,
        "source_families": list(source_families),
        "target_family": target_family,
        "family_specs": deepcopy(FAMILY_SPECS),
        "reference_state": "displayed-order 0101... Neel",
        "staggered_z_convention": (
            "B_Zs = -sum_i (-1)^i Z_i, so positive coefficients favor the "
            "fixed 0101... Neel reference"
        ),
        "bit_order": "array/string position 0 is logical qubit 0 and the integer MSB",
        "time_rule": "t_d = d*pi/coefficient_l1; model tau_d = coefficient_l1*t_d = d*pi",
        "measurement": "computational-basis full N-bit bitstrings",
        "evolution": "exact sparse expm_multiply; no Trotter approximation",
        "model_receives_family_label": False,
        "evaluation_regime": evaluation_regime,
        "target_support_is_subset_of_source_union": True,
        "target_template_seen_in_source": target_template_seen,
        "instances": instances,
    }
    protocol["protocol_sha256"] = protocol_digest(protocol)
    return protocol


def protocol_digest(protocol: Mapping) -> str:
    payload = deepcopy(dict(protocol))
    payload.pop("protocol_sha256", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_protocol(protocol: Mapping) -> bool:
    if int(protocol.get("protocol_version", -1)) not in SUPPORTED_PROTOCOL_VERSIONS:
        raise ValueError("unsupported cross-family protocol version")
    if protocol.get("protocol_sha256") != protocol_digest(protocol):
        raise ValueError("protocol digest mismatch")
    if protocol.get("model_receives_family_label") is not False:
        raise ValueError("family-label leakage is forbidden")
    instances = protocol.get("instances", [])
    identifiers = [row["instance_id"] for row in instances]
    if not identifiers or len(identifiers) != len(set(identifiers)):
        raise ValueError("instance identifiers must be non-empty and unique")
    target_family = str(protocol.get("target_family", TARGET_FAMILY))
    source_families = set(protocol.get("source_families", SOURCE_FAMILIES))
    target_template_seen = bool(protocol.get("target_template_seen_in_source", False))
    if target_family in source_families and not target_template_seen:
        raise ValueError("target family appears among source families")
    if target_template_seen and target_family not in source_families:
        raise ValueError("in-family control target is absent from source families")
    if (
        not target_template_seen
        and any(
            row["family"] == target_family and row["split"] != "target_test"
            for row in instances
        )
    ):
        raise ValueError("target family leaked outside target_test")
    if any(row["family"] != target_family and row["split"] == "target_test" for row in instances):
        raise ValueError("source family leaked into target_test")
    if any(
        row["split"] in ("source_train", "source_holdout")
        and row["family"] not in source_families
        for row in instances
    ):
        raise ValueError("unknown family appears in a source split")
    return True
