# Algorithm Tests

This directory contains tests for the optimization algorithms in the `ao_shaping.algorithm` module.

## Running Tests

To run all algorithm tests:

```bash
pytest tests/ao_shaping/algorithm/
```

To run tests for a specific optimizer (e.g., AdaMOD):

```bash
pytest tests/ao_shaping/algorithm/test_adam.py::TestAdaMOD
```

To run a specific test case:

```bash
pytest tests/ao_shaping/algorithm/test_adam.py::TestAdaMOD::test_initialization
```

## Test Coverage

The tests cover the following optimizers:
- SGD
- Adam
- AdamW
- AdaMOD
- Muno (coming soon)
- MunoW (coming soon)
- Muon (coming soon)
- AdamNS (coming soon)

Each optimizer is tested for:
1. Proper initialization with default and custom parameters
2. Correct update behavior with various gradient inputs
3. Edge cases (zero gradients, large gradients, etc.)
4. Specific behavior unique to each optimizer