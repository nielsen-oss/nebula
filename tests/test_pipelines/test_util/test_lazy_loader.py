"""Tests for nested lazy parameter resolution.

These tests verify that lazy references (functions and storage keys)
are correctly resolved at any nesting depth within parameter structures.
"""

from nebula.base import extract_lazy_params
from nebula.pipelines.pipeline_loader import (
    extract_lazy_params as extract_lazy_params_text,
)
from nebula.storage import nebula_storage as ns


class TestBasePyLazyResolution:
    """Tests for the Python API lazy parameter resolution."""

    @staticmethod
    def setup_method():
        ns.clear()

    def test_flat_ns_reference(self):
        """Flat (ns, "key") should be resolved."""
        ns.set("my_key", "stored_value")

        params = {"value": (ns, "my_key")}
        result = extract_lazy_params(params)

        assert result == {"value": "stored_value"}

    def test_flat_static_value(self):
        """Static values should pass through unchanged."""
        params = {"value": "static", "number": 123}
        result = extract_lazy_params(params)

        assert result == {"value": "static", "number": 123}

    # -------------------------------------------------------------------------
    # Nested parameter tests (NEW behavior)
    # -------------------------------------------------------------------------

    def test_nested_ns_in_dict(self):
        """(ns, "key") nested in dict should be resolved."""
        ns.set("threshold", 0.5)

        params = {"config": {"threshold": (ns, "threshold"), "static": True}}
        result = extract_lazy_params(params)

        assert result == {"config": {"threshold": 0.5, "static": True}}

    def test_nested_ns_in_list(self):
        """(ns, "key") nested in list should be resolved."""
        ns.set("col_name", "user_id")

        params = {"columns": ["static_col", (ns, "col_name")]}
        result = extract_lazy_params(params)

        assert result == {"columns": ["static_col", "user_id"]}

    def test_nested_ns_in_list_of_dicts(self):
        """(ns, "key") in list of dicts should be resolved - the AddLiterals case."""
        ns.set("dynamic_value", 999)

        params = {
            "data": [
                {"alias": "c1", "value": "static"},
                {"alias": "c2", "value": (ns, "dynamic_value")},
            ]
        }
        result = extract_lazy_params(params)

        assert result == {
            "data": [
                {"alias": "c1", "value": "static"},
                {"alias": "c2", "value": 999},
            ]
        }

    def test_deeply_nested_resolution(self):
        """Lazy references at arbitrary depth should be resolved."""
        ns.set("deep_value", "found_it")

        params = {
            "level1": {
                "level2": {
                    "level3": [
                        {"level4": (ns, "deep_value")},
                    ]
                }
            }
        }
        result = extract_lazy_params(params)

        assert result == {
            "level1": {
                "level2": {
                    "level3": [
                        {"level4": "found_it"},
                    ]
                }
            }
        }

    def test_tuple_preserved_when_not_ns_reference(self):
        """Regular tuples (not ns references) should be preserved."""
        params = {
            "bounds": (0, 100),  # Regular tuple, not ns reference
            "point": (1, 2, 3),
        }
        result = extract_lazy_params(params)

        assert result == {"bounds": (0, 100), "point": (1, 2, 3)}

    def test_mixed_static_and_lazy(self):
        """Mix of static and lazy values should work correctly."""
        ns.set("key1", "value1")
        ns.set("key2", "value2")

        params = {
            "static_str": "hello",
            "static_int": 123,
            "lazy_ns": (ns, "key1"),
            "nested": {"static": True, "lazy": (ns, "key2")},
            "list_mixed": [
                "static",
                (ns, "key1"),
            ],
        }
        result = extract_lazy_params(params)

        assert result == {
            "static_str": "hello",
            "static_int": 123,
            "lazy_ns": "value1",
            "nested": {"static": True, "lazy": "value2"},
            "list_mixed": ["static", "value1"],
        }


# =============================================================================
# Tests for pipeline_loader.py (YAML/JSON API)
# =============================================================================


class TestPipelineLoaderLazyResolution:
    """Tests for the YAML/JSON lazy parameter resolution."""

    @staticmethod
    def setup_method():
        ns.clear()

    def test_flat_ns_string_marker(self):
        """Flat __ns__ marker should become (ns, key) tuple."""
        params = {"value": "__ns__my_key"}
        result = extract_lazy_params_text(params)
        assert result["value"] == (ns, "my_key")

    # -------------------------------------------------------------------------
    # Nested parameter tests (NEW behavior)
    # -------------------------------------------------------------------------

    def test_nested_ns_in_dict(self):
        """__ns__ nested in dict should become (ns, key) tuple."""
        params = {"config": {"threshold": "__ns__threshold_key"}}
        result = extract_lazy_params_text(params)

        assert result == {"config": {"threshold": (ns, "threshold_key")}}

    def test_nested_ns_in_list_of_dicts(self):
        """__ns__ in list of dicts - the YAML AddLiterals case."""
        params = {
            "data": [
                {"alias": "c1", "value": "static"},
                {"alias": "c2", "value": "__ns__dynamic_value"},
            ]
        }
        result = extract_lazy_params_text(params)

        # After YAML parsing, we have (ns, key) tuples
        assert result["data"][0] == {"alias": "c1", "value": "static"}
        assert result["data"][1]["alias"] == "c2"
        assert result["data"][1]["value"] == (ns, "dynamic_value")

    def test_deeply_nested_markers(self):
        """Markers at arbitrary depth should be converted."""

        params = {
            "level1": {
                "level2": [
                    {"value": "__ns__deep_key"},
                ]
            }
        }
        result = extract_lazy_params_text(params)

        assert result["level1"]["level2"][0]["value"] == (ns, "deep_key")

    def test_static_strings_preserved(self):
        """Strings without markers should pass through unchanged."""
        params = {"name": "my_column", "nested": {"description": "some text"}}
        result = extract_lazy_params_text(params)

        assert result == params


# =============================================================================
# Integration tests (full flow)
# =============================================================================


class TestFullLazyFlow:
    """Test the complete lazy evaluation flow from definition to resolution."""

    @staticmethod
    def setup_method():
        ns.clear()

    def test_yaml_to_runtime_resolution(self):
        """
        Simulate the full flow:
        1. YAML config with __ns__ markers
        2. Converted to (ns, key) tuples by pipeline_loader
        3. Resolved to actual values by LazyWrapper at transform time
        """
        # Step 1: Simulate YAML config
        yaml_params = {
            "data": [
                {"alias": "static_col", "value": "hello"},
                {"alias": "dynamic_col", "value": "__ns__runtime_value"},
            ]
        }

        # Step 2: Pipeline loader converts markers to tuples
        loader_result = extract_lazy_params_text(yaml_params)

        # Verify intermediate state
        assert loader_result["data"][1]["value"] == (ns, "runtime_value")

        # Step 3: At transform time, storage has been populated
        ns.set("runtime_value", 42)

        # Step 4: LazyWrapper resolves the tuples
        runtime_result = extract_lazy_params(loader_result)

        # Verify final state
        assert runtime_result == {
            "data": [
                {"alias": "static_col", "value": "hello"},
                {"alias": "dynamic_col", "value": 42},
            ]
        }

    def test_add_literals_use_case(self):
        """
        Test the specific AddLiterals use case that motivated this change.

        YAML:
        - transformer: "AddLiterals"
          lazy: True
          params:
            data:
              - { "alias": "c5", "value": "__ns__my_key" }
        """
        # Simulate earlier pipeline step storing a value
        ns.set("computed_result", {"nested": "data", "count": 100})

        # YAML-style params
        yaml_params = {
            "data": [
                {"alias": "c3", "value": "static_literal"},
                {"alias": "c5", "value": "__ns__computed_result"},
            ]
        }

        # Pipeline loader processes
        loaded_params = extract_lazy_params_text(yaml_params)

        # LazyWrapper resolves at transform time
        resolved_params = extract_lazy_params(loaded_params)

        assert resolved_params == {
            "data": [
                {"alias": "c3", "value": "static_literal"},
                {"alias": "c5", "value": {"nested": "data", "count": 100}},
            ]
        }
