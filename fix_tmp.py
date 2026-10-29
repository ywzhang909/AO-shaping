from pathlib import Path
p = Path('src/ao_shaping/utils/image/target/objective.py')
t = p.read_text()
if 'from ao_shaping.utils.io.record_mode import RecordMode' not in t:
    t = t.replace('import numpy as np\n\n', 'import numpy as np\n\nfrom ao_shaping.utils.io.record_mode import RecordMode\n\n')
add = '''
MAXIMIZE_OBJECTIVES = ("pib", "avg_radiu", "shape", "roi_pib", "rms_pib")
MINIMIZE_OBJECTIVES = (
    "rms",
    "j2",
    "strehl",
    "eee",
    "sharpeness",
)
_MAXIMIZE_SET = frozenset(MAXIMIZE_OBJECTIVES)


def objective_direction(objective: str) -> RecordMode:
    obj = str(objective).strip().lower()
    if obj in _MAXIMIZE_SET:
        return RecordMode.MAX
    return RecordMode.MIN


def objective_maximizes(objective: str) -> bool:
    return objective_direction(objective) is RecordMode.MAX


'''
if 'objective_direction' not in t:
    t = t.replace('SHAPING_OBJECTIVE_CHOICES = ', add + 'SHAPING_OBJECTIVE_CHOICES = ')
p.write_text(t)
