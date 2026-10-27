\ = 'src/ao_shaping/utils/image/target/__init__.py'
\ = Get-Content \ -Raw
\ = "from .objective import (\n    SHAPING_OBJECTIVE_CHOICES,\n    ShapingObjective,\n    ShapingObjectiveParams,\n    TargetShape,\n)"
\ = "from .objective import (\n    MAXIMIZE_OBJECTIVES,\n    MINIMIZE_OBJECTIVES,\n    SHAPING_OBJECTIVE_CHOICES,\n    ShapingObjective,\n    ShapingObjectiveParams,\n    TargetShape,\n    objective_direction,\n    objective_maximizes,\n)"
if (\ -match [regex]::Escape(\)) { \ = \.Replace(\, \); Set-Content \ -Value \ -NoNewline; Write-Output 'done' } else { Write-Output 'no' }
