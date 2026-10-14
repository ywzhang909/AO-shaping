# AO-Shaping ML Training Helper Script
# This script sets PYTHONPATH automatically
$env:PYTHONPATH = "src;libs"
Set-Location $PSScriptRoot

# Pass all arguments to the training script
uv run python src/ao_shaping/ml/train.py $args