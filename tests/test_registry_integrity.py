from __future__ import annotations

from models.catalog import (
    ALGORITHM_CATALOG,
    catalog_executable_model_ids,
    executable_model_ids,
    validate_catalog_registry_integrity,
    validate_model_contracts,
    model_contracts,
)
from tools.model_registry import (
    ADVANCED_MODEL_REGISTRY,
    BASIC_MODEL_REGISTRY,
    MODEL_SWEEP_CONFIGS,
    registered_model_ids,
    registry_table_suffixes,
    validate_model_registry,
)


def test_model_registry_has_no_internal_integrity_errors():
    assert validate_model_registry() == []


def test_every_registered_model_has_complete_contract():
    assert validate_model_contracts() == []
    contracts = model_contracts()
    assert set(contracts) == registered_model_ids()
    assert all(contract.metrics for contract in contracts.values())
    assert all(contract.assumptions for contract in contracts.values())


def test_catalog_and_codegen_registry_are_consistent():
    assert validate_catalog_registry_integrity(import_symbols=True) == []


def test_executable_labels_match_codegen_registry():
    assert executable_model_ids() == registered_model_ids()


def test_catalog_executable_ids_are_labeled_and_registered():
    expected = {
        model_id
        for entry in ALGORITHM_CATALOG
        for model_id in entry.executable_model_ids
    }

    assert catalog_executable_model_ids() == expected
    assert catalog_executable_model_ids().issubset(executable_model_ids())
    assert catalog_executable_model_ids().issubset(registered_model_ids())


def test_registry_auxiliary_configs_reference_registered_models():
    assert set(MODEL_SWEEP_CONFIGS).issubset(registered_model_ids())
    assert registry_table_suffixes()
    assert "" not in registry_table_suffixes()
    assert registered_model_ids() == (
        set(BASIC_MODEL_REGISTRY)
        | {model_id for model_id, *_ in ADVANCED_MODEL_REGISTRY}
    )
