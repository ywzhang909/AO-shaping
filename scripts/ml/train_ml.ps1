# AO-Shaping ML Training Helper Script
# This script sets PYTHONPATH automatically
$env:PYTHONPATH = "src;libs"
Set-Location "$PSScriptRoot\..\.."

# Pass all arguments to the training script
uv run python -m ml.train $args