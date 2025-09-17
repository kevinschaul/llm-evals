#!/usr/bin/env python
"""Tests for eval.py"""

import pytest
import tempfile
import json
import csv
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import yaml
from argparse import ArgumentParser

# Import the modules we're testing
from eval import (
    cli, load_config, parse_boolean, strip_think_tags, to_uppercase,
    apply_transforms, levenshtein_distance, run_assertions, aggregate_results,
    CompletedResult, check_assertion
)


class TestCLI:
    """Test CLI argument parsing"""
    
    def test_basic_args(self):
        """Test basic argument parsing"""
        with patch('sys.argv', ['eval.py', 'config.yaml']):
            args = cli()
            assert args.eval_config == 'config.yaml'
            assert args.max_per_provider is None
            assert args.provider is None

    def test_max_per_provider_flag(self):
        """Test --max-per-provider flag"""
        with patch('sys.argv', ['eval.py', 'config.yaml', '--max-per-provider', '5']):
            args = cli()
            assert args.max_per_provider == 5

    def test_provider_flag(self):
        """Test --provider flag"""
        with patch('sys.argv', ['eval.py', 'config.yaml', '--provider', 'gpt-4']):
            args = cli()
            assert args.provider == 'gpt-4'


class TestConfigLoading:
    """Test configuration loading and processing"""

    def test_load_basic_yaml_config(self):
        """Test loading basic YAML config"""
        config_data = {
            'description': 'Test eval',
            'providers': [{'id': 'test-provider', 'model': 'gpt-4'}],
            'prompts': ['Test prompt'],
            'tests': [{'vars': {'input': 'hello'}, 'expected': 'hi'}]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            f.flush()
            
            config = load_config(f.name)
            
            assert config['description'] == 'Test eval'
            assert len(config['providers']) == 1
            assert config['providers'][0]['id'] == 'test-provider'
            assert len(config['prompts']) == 1
            assert config['prompts'][0]['id'] == 'prompt-1'
            assert config['prompts'][0]['text'] == 'Test prompt'
            assert len(config['tests']) == 1
            assert config['tests'][0]['id'] == 'test-1'
            assert config['tests'][0]['vars']['input'] == 'hello'
            assert config['tests'][0]['expected'] == 'hi'

    def test_load_csv_tests(self):
        """Test loading tests from CSV file"""
        # Create temporary CSV file
        csv_data = [
            ['input', 'context', '__expected'],
            ['hello', 'greeting', 'hi'],
            ['goodbye', 'farewell', 'bye']
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / 'tests.csv'
            config_path = Path(temp_dir) / 'config.yaml'
            
            # Write CSV
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(csv_data)
            
            # Write config that references CSV
            config_data = {
                'description': 'Test eval',
                'providers': [{'id': 'test-provider', 'model': 'gpt-4'}],
                'prompts': ['Test {{input}}'],
                'tests': 'file://tests.csv'
            }
            
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f)
            
            config = load_config(str(config_path))
            
            assert len(config['tests']) == 2
            assert config['tests'][0]['id'] == 'test-1'
            assert config['tests'][0]['vars']['input'] == 'hello'
            assert config['tests'][0]['vars']['context'] == 'greeting'
            assert config['tests'][0]['vars']['__expected'] == 'hi'
            assert config['tests'][0]['expected'] == 'hi'

    def test_csv_assertion_types(self):
        """Test different CSV assertion types"""
        csv_data = [
            ['input', '__expected'],
            ['test1', 'exact_match'],
            ['test2', 'icontains:Case_Insensitive'],
            ['test3', 'contains:substring'],
            ['test4', 'levenshtein(2):fuzzy_match'],
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / 'tests.csv'
            config_path = Path(temp_dir) / 'config.yaml'
            
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(csv_data)
            
            config_data = {
                'providers': [{'id': 'test'}],
                'prompts': ['{{input}}'],
                'tests': 'file://tests.csv'
            }
            
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f)
            
            config = load_config(str(config_path))
            
            # Test exact match assertion
            assert config['tests'][0]['expected'] == 'exact_match'
            
            # Test icontains assertion
            assert config['tests'][1]['expected'] == 'icontains:Case_Insensitive'
            
            # Test contains assertion
            assert config['tests'][2]['expected'] == 'contains:substring'
            
            # Test levenshtein assertion
            assert config['tests'][3]['expected'] == 'levenshtein(2):fuzzy_match'

    def test_inline_yaml_with_file_reference(self):
        """Test inline YAML tests with file:// references in expected"""
        with tempfile.TemporaryDirectory() as temp_dir:
            expected_path = Path(temp_dir) / 'expected.json'
            config_path = Path(temp_dir) / 'config.yaml'
            
            # Write expected file
            expected_content = {'result': 'success', 'count': 5}
            with open(expected_path, 'w') as f:
                json.dump(expected_content, f)
            
            # Write config with inline YAML tests
            config_data = {
                'providers': [{'id': 'test-provider', 'model': 'gpt-4'}],
                'prompts': ['Process {{input}}'],
                'tests': [
                    {
                        'vars': {
                            'input': 'test_data',
                            'transforms': ['parse_boolean'],
                            'attachments': [{'path': 'test.pdf'}]
                        },
                        'expected': 'file://expected.json'
                    }
                ]
            }
            
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f)
            
            config = load_config(str(config_path))
            
            # Check that file:// reference was resolved
            assert config['tests'][0]['expected'] == json.dumps(expected_content)
            assert config['tests'][0]['vars']['input'] == 'test_data'
            assert config['tests'][0]['vars']['transforms'] == ['parse_boolean']


class TestTransforms:
    """Test text transform functions"""

    def test_parse_boolean_json_response(self):
        """Test parsing JSON response with category field"""
        assert parse_boolean('{"category": true}') is True
        assert parse_boolean('{"category": false}') is False

    def test_parse_boolean_simple_values(self):
        """Test parsing simple true/false values"""
        assert parse_boolean('true') is True
        assert parse_boolean('false') is False
        assert parse_boolean('True') is True
        assert parse_boolean('False') is False

    def test_parse_boolean_quoted_values(self):
        """Test parsing quoted true/false values"""
        assert parse_boolean('"true"') is True
        assert parse_boolean("'false'") is False

    def test_parse_boolean_pattern_matching(self):
        """Test parsing TRUE/FALSE patterns in text"""
        assert parse_boolean('The answer is TRUE') is True
        assert parse_boolean('The answer is FALSE') is False

    def test_parse_boolean_fallback(self):
        """Test fallback when parsing fails"""
        result = parse_boolean('ambiguous text')
        assert result == 'ambiguous text'  # Should return original text

    def test_strip_think_tags(self):
        """Test removing think tags"""
        text = "Start <think>internal thoughts</think> end"
        assert strip_think_tags(text) == "Start  end"
        
        text = "No tags here"
        assert strip_think_tags(text) == "No tags here"
        
        text = "<think>first</think> middle <think>second</think>"
        assert strip_think_tags(text) == "middle"

    def test_to_uppercase(self):
        """Test uppercase transform"""
        assert to_uppercase("hello world") == "HELLO WORLD"
        assert to_uppercase("MiXeD cAsE") == "MIXED CASE"

    def test_apply_transforms_chain(self):
        """Test applying multiple transforms in sequence"""
        text = "<think>ignore this</think> The answer is true"
        transforms = ["strip_think_tags", "parse_boolean"]
        result = apply_transforms(text, transforms)
        assert result is True

    def test_apply_transforms_unknown(self):
        """Test handling unknown transforms"""
        with patch('eval.logger.warning') as mock_warning:
            result = apply_transforms("test", ["unknown_transform"])
            assert result == "test"
            mock_warning.assert_called_with("Unknown transform: unknown_transform")


class TestAssertions:
    """Test assertion evaluation"""

    def test_equals_assertion(self):
        """Test equals assertion"""
        assert check_assertion("expected_value", "expected_value") is True
        assert check_assertion("different_value", "expected_value") is False
        
        # Test with run_assertions wrapper
        assert run_assertions("expected_value", "expected_value") is True
        assert run_assertions("different_value", "expected_value") is False

    def test_contains_assertion(self):
        """Test contains assertion"""
        assert check_assertion("this contains substring", "contains:substring") is True
        assert check_assertion("this does not", "contains:substring") is False

    def test_icontains_assertion(self):
        """Test case-insensitive contains assertion"""
        assert check_assertion("this contains substring", "icontains:SUBSTRING") is True
        assert check_assertion("CONTAINS SUBSTRING", "icontains:substring") is True
        assert check_assertion("no match", "icontains:SUBSTRING") is False

    def test_levenshtein_assertion(self):
        """Test Levenshtein distance assertion"""
        assert check_assertion("hello", "levenshtein(2):hello") is True  # distance 0
        assert check_assertion("helo", "levenshtein(2):hello") is True   # distance 1
        assert check_assertion("hllo", "levenshtein(2):hello") is True   # distance 1
        assert check_assertion("help", "levenshtein(2):hello") is True   # distance 2
        assert check_assertion("world", "levenshtein(2):hello") is False # distance > 2

    def test_boolean_type_conversion(self):
        """Test boolean type conversion in assertions"""
        assert check_assertion(True, "true") is True
        assert check_assertion(False, "true") is False
        assert check_assertion("true", True) is True
        assert check_assertion("false", True) is False

    def test_no_assertions_returns_none(self):
        """Test that no assertions returns None"""
        assert run_assertions("any_result", None) is None
        assert run_assertions("any_result", "") is None


class TestUtilities:
    """Test utility functions"""

    def test_levenshtein_distance(self):
        """Test Levenshtein distance calculation"""
        assert levenshtein_distance("", "") == 0
        assert levenshtein_distance("hello", "hello") == 0
        assert levenshtein_distance("hello", "helo") == 1
        assert levenshtein_distance("hello", "world") == 4
        assert levenshtein_distance("kitten", "sitting") == 3

    def test_aggregate_results(self):
        """Test result aggregation"""
        results = [
            CompletedResult(
                provider_id="gpt-4", prompt_id="prompt-1", test="test-1",
                prompt="Test prompt", result="result1", error=None,
                duration_ms=100, passed=True, expected="expected1",
                timestamp="2023-01-01T00:00:00Z"
            ),
            CompletedResult(
                provider_id="gpt-4", prompt_id="prompt-1", test="test-2",
                prompt="Test prompt", result="result2", error=None,
                duration_ms=200, passed=False, expected="expected2",
                timestamp="2023-01-01T00:00:01Z"
            ),
            CompletedResult(
                provider_id="gpt-4", prompt_id="prompt-1", test="test-3",
                prompt="Test prompt", result="result3", error=None,
                duration_ms=150, passed=None, expected="expected3",
                timestamp="2023-01-01T00:00:02Z"
            ),
        ]
        
        aggregated = aggregate_results(results)
        
        assert len(aggregated) == 1
        agg = aggregated[0]
        assert agg["provider_id"] == "gpt-4"
        assert agg["prompt_id"] == "prompt-1"
        assert agg["total_tests"] == 3
        assert agg["passed"] == 1
        assert agg["failed"] == 1
        assert agg["no_assertions"] == 1
        assert agg["pass_rate"] == 50.0  # 1 passed out of 2 with assertions
        assert agg["errors"] == 0
        assert agg["avg_duration_ms"] == 150.0  # (100 + 200 + 150) / 3


class TestIntegration:
    """Integration tests that test multiple components together"""

    @patch('eval.llm.get_model')
    @patch('eval.sqlite_utils.Database')
    def test_cli_to_config_loading(self, mock_db, mock_get_model):
        """Test that CLI args correctly influence config processing"""
        # This would be a more complex integration test
        # that tests the flow from CLI parsing through config loading
        # to actual execution - but it requires more complex mocking
        pass


if __name__ == "__main__":
    pytest.main([__file__])
