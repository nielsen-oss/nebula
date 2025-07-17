"""Unittests for 'loop_exploder' module."""

from copy import deepcopy
from itertools import product

import pytest

from nlsn.nebula.pipelines.loop_exploder import (
    _explode_loops,
    convert_product_to_linear,
    explode_loops_in_pipeline,
    prepare_loop_params,
    process_loop,
    substitute_params,
    validate_loop_params,
)
from nlsn.nebula.pipelines.pipeline_loader import load_pipeline


class TestLoopValidation:
    @staticmethod
    @pytest.mark.parametrize("gen_type", [None, "linear", "product"])
    def test_validate_loop_params_valid(gen_type):
        """Test validation with valid linear loop parameters."""
        d = {
            "values": {"names": ["a", "b", "c"], "numbers": [1, 2, 3]},
        }
        if gen_type:
            d["generation_type"] = gen_type
        validate_loop_params(d)

    @staticmethod
    @pytest.mark.parametrize(
        "gen_type, err", [(123, TypeError), ("invalid", ValueError)]
    )
    def test_validate_loop_params_invalid_generation_type(gen_type, err):
        """Test validation fails with invalid generation_type."""
        d = {
            "values": {"names": ["a", "b"], "numbers": [1, 2]},
            "generation_type": gen_type,
        }
        with pytest.raises(err):
            validate_loop_params(d)

    @staticmethod
    def test_validate_loop_params_missing_values():
        """Test validation fails when 'values' key is missing."""
        d = {"generation_type": "linear"}
        with pytest.raises(KeyError):
            validate_loop_params(d)

    @staticmethod
    @pytest.mark.parametrize(
        "values, err",
        [
            ({"names": ["a", "b"], "numbers": "not_a_list"}, TypeError),
            ({"names": ["a", "b", "c"], "numbers": [1, 2]}, ValueError),
            ({}, ValueError),
        ],
    )
    def test_validate_loop_params_invalid_values(values, err):
        """Test validation fails when values are not valid."""
        d = {"values": values}
        with pytest.raises(err):
            validate_loop_params(d)

    @staticmethod
    @pytest.mark.parametrize("gen_type", [None, "linear", "product"])
    def test_convert_product_to_linear(gen_type):
        """Test conversion function for linear generation."""
        d = {
            "values": {"names": ["a", "b"], "numbers": [1, 2]},
        }
        if gen_type is not None:
            d["generation_type"] = gen_type
        result = convert_product_to_linear(d)
        if gen_type != "product":
            assert result == {k: v for k, v in d.items() if k != "generation_type"}
            return

        exp = {"values": {"names": ("a", "a", "b", "b"), "numbers": (1, 2, 1, 2)}}
        assert result == exp

    @staticmethod
    def test_convert_product_to_linear_product_conversion():
        """Test conversion from product to linear."""
        d = {
            "values": {"names": ["a", "b"], "numbers": [1, 2]},
            "generation_type": "product",
        }
        result = convert_product_to_linear(d)

        # Check the structure
        assert "values" in result
        assert "generation_type" not in result

        # Check the values are correctly expanded
        expected_values = {"names": ("a", "a", "b", "b"), "numbers": (1, 2, 1, 2)}
        assert result["values"] == expected_values

    @staticmethod
    def test_convert_product_to_linear_complex_product():
        """Test conversion with three variables."""
        d = {
            "values": {"names": ["a", "b"], "numbers": [1, 2], "letters": ["x", "y"]},
            "generation_type": "product",
        }
        result = convert_product_to_linear(d)

        # Check the result length (should be 2 * 2 * 2 = 8)
        assert len(next(iter(result["values"].values()))) == 8

        # Verify all combinations are present
        all_combinations = list(product(["a", "b"], [1, 2], ["x", "y"]))
        for i, comb in enumerate(all_combinations):
            assert result["values"]["names"][i] == comb[0]
            assert result["values"]["numbers"][i] == comb[1]
            assert result["values"]["letters"][i] == comb[2]

    @staticmethod
    def test_convert_product_to_linear_preserves_other_keys():
        """Test that conversion preserves non-values keys."""
        d = {
            "values": {"names": ["a", "b"], "numbers": [1, 2]},
            "generation_type": "product",
            "other_key": "should_be_preserved",
            "another_key": [1, 2, 3],
        }
        result = convert_product_to_linear(d)
        assert result["other_key"] == "should_be_preserved"
        assert result["another_key"] == [1, 2, 3]

    @staticmethod
    @pytest.mark.parametrize("gen_type", [None, "linear", "product"])
    def test_prepare_loop_params(gen_type):
        """Test the outer function 'prepare_loop_params'."""
        d = {
            "values": {"names": ["a", "b"], "numbers": [1, 2]},
            "generation_type": gen_type,
        }
        prepare_loop_params(d)


class TestSubstituteParams:
    @staticmethod
    def test_simple_string():
        """Test simple string parameter substitution."""
        d = {"name": "Hello <<user>>!"}
        params = {"user": "John"}
        result = substitute_params(d, params)
        assert result == {"name": "Hello John!"}

    @staticmethod
    def test_multiple_in_string():
        """Test multiple parameters in a single string."""
        d = {"greeting": "Hello <<first>> <<last>>!"}
        params = {"first": "John", "last": 2}
        result = substitute_params(d, params)
        assert result == {"greeting": "Hello John 2!"}

    @staticmethod
    def test_nested_dict():
        """Test substitution in a nested dictionary."""
        d = {
            "outer": {
                "inner": "Value: <<value1>>",
                "other": "<<other_value>>",
                "list": "<<list>>",
                "list_str": "list_str_<<list>>",
            }
        }
        params = {"value1": "42", "other_value": -999, "list": [1, 2]}
        result = substitute_params(d, params)
        assert result == {
            "outer": {
                "inner": "Value: 42",
                "other": -999,
                "list": [1, 2],
                "list_str": f"list_str_{[1, 2]}",
            }
        }

    @staticmethod
    def test_list():
        """Test substitution in list values."""
        d = {"items": ["<<item1>>", "<<item2>>", "static"]}
        params = {"item1": "first", "item2": "second"}
        result = substitute_params(d, params)
        assert result == {"items": ["first", "second", "static"]}

    @staticmethod
    def test_list_of_dicts():
        """Test substitution in a list of dictionaries."""
        d = {"items": [{"name": "<<name1>>"}, {"name": "<<name2>>"}]}
        params = {"name1": "First", "name2": "Second"}
        result = substitute_params(d, params)
        assert result == {"items": [{"name": "First"}, {"name": "Second"}]}

    @staticmethod
    def test_no_substitution_needed():
        """Test when no substitution is needed."""
        d = {"static": "value", "nested": {"also_static": 123}, "list": ["a", "b", "c"]}
        params = {"param": "unused"}
        result = substitute_params(d, params)
        assert result == d
        assert result is not d  # Should be a new copy

    @staticmethod
    def test_missing_param():
        """Test behavior when a parameter is missing."""
        d = {"message": "Hello <<name>>!"}
        params = {"other": "value"}
        assert d == substitute_params(d, params)

    @staticmethod
    def test_number_conversion():
        """Test substitution with non-string parameter values."""
        d = {"message": "Value is <<value>>", "items": ["Count: <<count>>"]}
        params = {"value": 42, "count": 123}
        result = substitute_params(d, params)
        assert result == {"message": "Value is 42", "items": ["Count: 123"]}

    @staticmethod
    def test_mixed_content():
        """Test complex nested structure with mixed content types."""
        d = {
            "string": "Hello <<name>>",
            "number": 42,
            "list": ["<<item1>>", {"nested": "<<item2>>"}, 123, ["<<item3>>"]],
            "dict": {"a": "<<value1>>", "b": {"c": "<<value2>>"}},
        }
        params = {
            "name": "John",
            "item1": "first",
            "item2": "second",
            "item3": "third",
            "value1": "val1",
            "value2": "val2",
        }
        expected = {
            "string": "Hello John",
            "number": 42,
            "list": ["first", {"nested": "second"}, 123, ["third"]],
            "dict": {"a": "val1", "b": {"c": "val2"}},
        }
        result = substitute_params(d, params)
        assert result == expected

    @staticmethod
    def test_empty_inputs():
        """Test with empty dictionary and params."""
        assert substitute_params({}, {}) == {}
        assert substitute_params({"key": "value"}, {}) == {"key": "value"}

    @staticmethod
    def test_partial_match():
        """Test when a string contains a partial parameter pattern."""
        d = {
            "partial": "Value <<param",
            "correct": "Value <<param>>",
            "other": ">> param",
        }
        params = {"param": "test"}
        result = substitute_params(d, params)
        assert result == {
            "partial": "Value <<param",
            "correct": "Value test",
            "other": ">> param",
        }


class TestProcessLoop:
    @staticmethod
    def test_simple():
        """Test basic loop processing with simple values."""
        loop_dict = {
            "values": {"names": ["a", "b", "c"], "numbers": [1, 2, 3]},
            "template": "<<names>>_<<numbers>>",
        }

        result = process_loop(loop_dict)
        assert len(result) == 3
        assert all(isinstance(item, dict) for item in result)
        assert "template" in result[0]

    @staticmethod
    def test_empty_values():
        """Test processing with an empty values list."""
        loop_dict = {
            "values": {"names": [], "numbers": []},
            "template": "<<names>>_<<numbers>>",
        }

        result = process_loop(loop_dict)
        assert not result

    @staticmethod
    def test_single_iteration():
        """Test processing with a single iteration."""
        loop_dict = {
            "values": {"name": ["test"], "value": [42]},
            "template": "<<name>>_<<value>>",
        }

        result = process_loop(loop_dict)
        assert len(result) == 1
        assert "template" in result[0]

    @staticmethod
    def test_preserves_non_loop_keys():
        """Test that non-loop keys are preserved in the output."""
        loop_dict = {
            "values": {"name": ["a", "b"]},
            "static_key": "static_value",
            "another_key": 42,
        }

        result = process_loop(loop_dict)
        assert len(result) == 2
        assert all("static_key" in item for item in result)
        assert all("another_key" in item for item in result)
        assert all(item["static_key"] == "static_value" for item in result)
        assert all(item["another_key"] == 42 for item in result)

    @staticmethod
    def test_complex_structure():
        """Test processing with a complex nested structure."""
        loop_dict = {
            "values": {"name": ["a", "b"], "value": [1, 2]},
            "nested": {"template": "<<name>>_<<value>>", "fixed": "static"},
            "list": ["<<name>>", "fixed", {"key": "<<value>>"}],
        }

        result = process_loop(loop_dict)
        assert len(result) == 2
        assert all("nested" in item for item in result)
        assert all("list" in item for item in result)
        assert all(item["nested"]["fixed"] == "static" for item in result)

    @staticmethod
    def test_parameter_combinations():
        """Test that all parameter combinations are correctly processed."""
        loop_dict = {
            "values": {"x": ["1", "2"], "y": ["a", "b"]},
            "template": "<<x>>_<<y>>",
        }

        result = process_loop(loop_dict)
        assert len(result) == 2

        # Check the first iteration
        assert result[0]["template"] == "1_a"
        # Check the second iteration
        assert result[1]["template"] == "2_b"

    @staticmethod
    def test_with_none_values():
        """Test processing with None values in parameters."""
        loop_dict = {
            "values": {"name": ["a", None], "value": [1, 2]},
            "template": "<<name>>_<<value>>",
        }

        result = process_loop(loop_dict)
        assert len(result) == 2
        assert "template" in result[0]
        assert "template" in result[1]

    @staticmethod
    @pytest.mark.parametrize(
        "invalid_dict",
        [
            {},  # Empty dict
            {"values": None},  # None values
            {"values": {}},  # Empty values dict
            {"values": {"key": None}},  # None list
        ],
    )
    def test_invalid_inputs(invalid_dict):
        """Test processing with various invalid inputs."""
        with pytest.raises((KeyError, TypeError, ValueError)):
            process_loop(invalid_dict)

    @staticmethod
    def test_large_dataset():
        """Test processing with a larger dataset to check performance and memory."""
        loop_dict = {
            "values": {
                "id": list(range(1000)),
                "name": [f"name_{i}" for i in range(1000)],
            },
            "template": "<<id>>_<<name>>",
        }

        result = process_loop(loop_dict)
        assert len(result) == 1000
        assert all("template" in item for item in result)

    @staticmethod
    def test_type_consistency():
        """Test that output maintains consistent types."""
        loop_dict = {
            "values": {"num": [1, 2], "str": ["a", "b"], "bool": [True, False]},
            "template": {
                "num_val": "<<num>>",
                "str_val": "<<str>>",
                "bool_val": "<<bool>>",
            },
        }

        result = process_loop(loop_dict)
        assert len(result) == 2
        assert all(isinstance(item["template"], dict) for item in result)


class TestExplodeLoops:
    @staticmethod
    def test_no_loops():
        """Test that dictionaries without loops are properly copied."""
        d = {
            "transformer": "MyTransformer",
            "params": {"a": 1, "b": "value"},
            "nested": {"x": 10, "y": [1, 2, 3]},
        }
        result, is_exploded = _explode_loops(d)
        assert not is_exploded
        assert result == d
        assert result is not d  # Should be a new dict
        assert result["nested"] is not d["nested"]  # Nested dict should be new
        assert result["nested"]["y"] == d["nested"]["y"]  # Lists should be equal

    @staticmethod
    def test_simple_loop():
        """Test processing of a simple loop structure."""
        d = {
            "loop": {
                "values": {"names": ["a", "b"], "numbers": [1, 2]},
                "transformer": "MyTransformer",
                "params": {"name": "<<names>>", "value": "<<numbers>>"},
            }
        }
        result, is_exploded = _explode_loops(d)
        assert is_exploded
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["params"]["name"] == "a"
        assert result[1]["params"]["name"] == "b"

    @staticmethod
    def test_nested_structures():
        """Test that nested structures are properly handled."""
        d = {
            "pipeline": [
                {"transformer": "T1"},
                {
                    "loop": {
                        "values": {"x": [1, 2]},
                        "transformer": "T2",
                        "params": {"value": "<<x>>"},
                    }
                },
            ]
        }
        result, is_exploded = _explode_loops(d)
        assert not is_exploded  # It explodes the innermost dict
        assert isinstance(result, dict)
        assert "pipeline" in result
        assert len(result["pipeline"]) >= 2

    @staticmethod
    def test_nested_loops():
        """Test handling of nested loops."""
        d = {
            "loop": {
                "values": {"outer": ["A", "B"]},
                "pipeline": [
                    {
                        "loop": {
                            "values": {"inner": [1, 2]},
                            "params": {
                                "outer_val": "<<outer>>",
                                "inner_val": "<<inner>>",
                            },
                        }
                    }
                ],
            }
        }
        result, is_exploded = _explode_loops(d)
        assert is_exploded
        assert isinstance(result, list)
        # Should have expanded both loops
        assert len(result) > 1
        # Check inner loop expansion
        for item in result:
            assert isinstance(item["pipeline"], list)

    @staticmethod
    def test_immutable_values():
        """Test that immutable values are handled correctly."""
        d = {
            "string": "value",
            "number": 42,
            "tuple": (1, 2, 3),
            "nested": {"immutable": "test"},
        }
        result, is_exploded = _explode_loops(d)
        assert not is_exploded
        assert result == d
        assert result is not d
        assert result["nested"] is not d["nested"]


class TestExplodeLoopsInPipeline:
    @staticmethod
    def test_simple():
        """Test processing of a simple pipeline."""
        pipe = {
            "pipeline": [{"transformer": "T1"}, {"transformer": "T2"}],
            "metadata": {"version": 1},
        }
        result = explode_loops_in_pipeline(pipe)
        assert result is not pipe
        assert result == pipe

    @staticmethod
    def test_with_loops():
        """Test pipeline processing with loops."""
        pipe = {
            "pipeline": [
                {"transformer": "T1"},
                {
                    "loop": {
                        "values": {"params": ["p1", "p2"]},
                        "transformer": "T2",
                        "params": {"value": "<<params>>"},
                    }
                },
            ]
        }
        result = explode_loops_in_pipeline(pipe)
        assert len(result["pipeline"]) > len(pipe["pipeline"])
        assert result["pipeline"][0] == pipe["pipeline"][0]

    @staticmethod
    def test_nested_structure():
        """Test processing of deeply nested pipeline structures."""
        pipe = {
            "pipeline": [
                {"pipeline": [{"transformer": "Inner1"}, {"transformer": "Inner2"}]},
                {"transformer": "Outer"},
            ]
        }
        result = explode_loops_in_pipeline(pipe)
        assert isinstance(result["pipeline"], list)
        assert len(result["pipeline"]) == len(pipe["pipeline"])

    @staticmethod
    def test_empty():
        """Test processing of an empty pipeline."""
        pipe = {"pipeline": [], "metadata": {"version": 1}}
        result = explode_loops_in_pipeline(pipe)
        assert isinstance(result["pipeline"], list)
        assert not result["pipeline"]
        assert result["metadata"] == pipe["metadata"]

    @staticmethod
    def test_no_pipeline_key():
        """Test processing when a pipeline key is missing."""
        pipe = {"pipeline": {"metadata": {"v": 1}, "config": {"param": "v"}}}
        result = explode_loops_in_pipeline(pipe)
        assert result == pipe
        assert result is not pipe

    @staticmethod
    def test_mutation_safety():
        """Test that original structures are not mutated."""
        original_pipe = {
            "pipeline": [
                {"transformer": "T1"},
                {"loop": {"values": {"x": [1, 2]}, "params": {"value": "<<x>>"}}},
            ]
        }
        pipe_copy = deepcopy(original_pipe)
        _result = explode_loops_in_pipeline(original_pipe)
        assert original_pipe == pipe_copy  # Original should not be modified

    @staticmethod
    def test_nested_dict_independence():
        """Test that nested dictionaries are independent in the result."""
        pipe = {"pipeline": [{"nested": {"deep": {"value": 42}}}]}
        result = explode_loops_in_pipeline(pipe)
        assert result["pipeline"][0]["nested"] is not pipe["pipeline"][0]["nested"]
        assert (
            result["pipeline"][0]["nested"]["deep"]
            is not pipe["pipeline"][0]["nested"]["deep"]
        )

    @staticmethod
    @pytest.mark.parametrize("invalid_input", [None, [], "string", 42])
    def test_invalid_inputs(invalid_input):
        """Test handling of invalid inputs."""
        with pytest.raises((TypeError, AttributeError)):
            explode_loops_in_pipeline(invalid_input)

    @staticmethod
    def test_actual_hardcoded_flat_pipeline():
        """Unit-test with a hardcoded nested flat pipeline."""
        pipe = {
            "pipeline": [
                {"transformer": "Distinct"},
                {
                    "pipeline": [
                        {
                            "transformer": "DuplicateColumn",
                            "params": {
                                "col_map": {
                                    "adev": "adev_orig",
                                    "local_time_category": "timeband",
                                }
                            },
                        },
                        {
                            "loop": {  # outermost loop
                                # "generation_type": "linear",
                                "values": {
                                    "algos": ["algo_X", "algo_Y"],
                                    "names": ["name_a", "name_b"],
                                    "pivot_col": [None, "timeband"],
                                },
                                "branch": {
                                    "end": "join",
                                    "how": "inner",
                                    "on": "adev",
                                    # This must become "name_a", "name_b"
                                    "storage": "<<names>>",
                                },
                                # This must become "branch_name_a", "branch_name_b"
                                "name": "branch_<<names>>",
                                "pipeline": [
                                    {
                                        "transformer": "WithColumn",
                                        "params": {
                                            # This must become: None, "timeband"
                                            "column_name": "<<pivot_col>>",
                                            "value": 3,
                                        },
                                    },
                                    {
                                        "loop": {  # innermost loop
                                            "generation_type": "product",
                                            "values": {
                                                "numbers": [20, 30],
                                                # This "names" must not be
                                                # overwritten by the outermost "names"
                                                "names": ["NEW_NAME"],
                                            },
                                            "transformer": "ReplaceWithMap",
                                            "params": {
                                                # This must become "algo_X", "algo_Y"
                                                # from the outermost loop
                                                "input_col": "<<algos>>",
                                                # This must become: None, "timeband"
                                                "replace": {"hello": "<<numbers>>"},
                                                # -> "col_A_1", "col_A_2", "col_A_3", "col_B_1", ...
                                                "output_col": "col_<<names>>_<<numbers>>",
                                            },
                                        }
                                    },
                                ],
                            }
                        },
                    ]
                },
            ]
        }

        exp = {
            "pipeline": [
                {"transformer": "Distinct"},
                {
                    "pipeline": [
                        {
                            "transformer": "DuplicateColumn",
                            "params": {
                                "col_map": {
                                    "adev": "adev_orig",
                                    "local_time_category": "timeband",
                                }
                            },
                        },
                        {
                            "branch": {
                                "end": "join",
                                "how": "inner",
                                "on": "adev",
                                "storage": "name_a",
                            },
                            "name": "branch_name_a",
                            "pipeline": [
                                {
                                    "transformer": "WithColumn",
                                    "params": {
                                        "column_name": None,
                                        "value": 3,
                                    },
                                },
                                {
                                    "transformer": "ReplaceWithMap",
                                    "params": {
                                        "input_col": "algo_X",
                                        "replace": {"hello": 20},
                                        "output_col": "col_NEW_NAME_20",
                                    },
                                },
                                {
                                    "transformer": "ReplaceWithMap",
                                    "params": {
                                        "input_col": "algo_X",
                                        "replace": {"hello": 30},
                                        "output_col": "col_NEW_NAME_30",
                                    },
                                },
                            ],
                        },
                        {
                            "branch": {
                                "end": "join",
                                "how": "inner",
                                "on": "adev",
                                "storage": "name_b",
                            },
                            "name": "branch_name_b",
                            "pipeline": [
                                {
                                    "transformer": "WithColumn",
                                    "params": {
                                        "value": 3,
                                        "column_name": "timeband",
                                    },
                                },
                                {
                                    "transformer": "ReplaceWithMap",
                                    "params": {
                                        "input_col": "algo_Y",
                                        "replace": {"hello": 20},
                                        "output_col": "col_NEW_NAME_20",
                                    },
                                },
                                {
                                    "transformer": "ReplaceWithMap",
                                    "params": {
                                        "input_col": "algo_Y",
                                        "replace": {"hello": 30},
                                        "output_col": "col_NEW_NAME_30",
                                    },
                                },
                            ],
                        },
                    ]
                },
            ]
        }
        chk = explode_loops_in_pipeline(pipe)
        assert chk == exp
        loaded = load_pipeline(chk)
        loaded.show_pipeline(add_transformer_params=True)

    @staticmethod
    def test_actual_hardcoded_split_pipeline():
        """Unit-test with a hardcoded nested split pipeline."""
        pipe = {
            "pipeline": [
                {"transformer": "Distinct"},
                {
                    "loop": {  # outermost loop
                        # "generation_type": "linear",
                        "values": {
                            "algos": ["algo_X", "algo_Y"],
                            "names": ["name_a", "name_b"],
                            "pivot_col": [None, "timeband"],
                        },
                        "split_function": "outer_func",
                        "pipeline": {
                            "outer_split_1": {
                                "transformer": "WithColumn",
                                "params": {
                                    # This must become: None, "timeband"
                                    "column_name": "<<pivot_col>>",
                                    "value": 1,
                                },
                            },
                            "outer_split_2": {
                                "pipeline": [
                                    {
                                        "transformer": "WithColumn",
                                        # This must become: "name_a", "name_b"
                                        "params": {
                                            "column_name": "<<names>>",
                                            "value": 2,
                                        },
                                    },
                                    {
                                        "loop": {  # innermost loop
                                            "generation_type": "product",
                                            "values": {
                                                "numbers": [20, 30],
                                                # This "names" must not be
                                                # overwritten by the outermost "names"
                                                "names": ["NEW_NAME"],
                                            },
                                            "transformer": "ReplaceWithMap",
                                            "params": {
                                                # This must become "algo_X", "algo_Y"
                                                # from the outermost loop
                                                "input_col": "<<algos>>",
                                                # This must become: None, "timeband"
                                                "replace": {"hello": "<<numbers>>"},
                                                # -> "col_A_20", "col_A_30", "col_B_20", "col_B_30"
                                                "output_col": "col_<<names>>_<<numbers>>",
                                            },
                                        },
                                    },
                                ],
                            },
                        },
                    }
                },
            ],
        }

        exp = {
            "pipeline": [
                {"transformer": "Distinct"},
                {
                    "split_function": "outer_func",
                    "pipeline": {
                        "outer_split_1": {
                            "transformer": "WithColumn",
                            "params": {"column_name": None, "value": 1},
                        },
                        "outer_split_2": {
                            "pipeline": [
                                {
                                    "transformer": "WithColumn",
                                    "params": {"column_name": "name_a", "value": 2},
                                },
                                {
                                    "transformer": "ReplaceWithMap",
                                    "params": {
                                        "input_col": "algo_X",
                                        "replace": {"hello": 20},
                                        "output_col": "col_NEW_NAME_20",
                                    },
                                },
                                {
                                    "transformer": "ReplaceWithMap",
                                    "params": {
                                        "input_col": "algo_X",
                                        "replace": {"hello": 30},
                                        "output_col": "col_NEW_NAME_30",
                                    },
                                },
                            ]
                        },
                    },
                },
                {
                    "split_function": "outer_func",
                    "pipeline": {
                        "outer_split_1": {
                            "transformer": "WithColumn",
                            "params": {"column_name": "timeband", "value": 1},
                        },
                        "outer_split_2": {
                            "pipeline": [
                                {
                                    "transformer": "WithColumn",
                                    "params": {"column_name": "name_b", "value": 2},
                                },
                                {
                                    "transformer": "ReplaceWithMap",
                                    "params": {
                                        "input_col": "algo_Y",
                                        "replace": {"hello": 20},
                                        "output_col": "col_NEW_NAME_20",
                                    },
                                },
                                {
                                    "transformer": "ReplaceWithMap",
                                    "params": {
                                        "input_col": "algo_Y",
                                        "replace": {"hello": 30},
                                        "output_col": "col_NEW_NAME_30",
                                    },
                                },
                            ]
                        },
                    },
                },
            ]
        }
        chk = explode_loops_in_pipeline(pipe)
        assert chk == exp
        extra_functions = {"outer_func": lambda x: x}
        loaded = load_pipeline(chk, extra_functions=extra_functions)
        loaded.show_pipeline(add_transformer_params=True)
