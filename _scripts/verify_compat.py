"""Runtime-verify: (1) is 'sim' importable? (2) which __all__ names actually fail."""
import importlib

# 1) sim.*
for m in ["sim", "sim.digitaltwin"]:
    try:
        importlib.import_module(m)
        print(f"[IMPORT-OK] {m}")
    except Exception as e:
        print(f"[IMPORT-FAIL] {m}: {type(e).__name__}: {e}")

# 2) runtime-check __all__ for the two dm __init__ files
def check_all(modpath, names):
    try:
        mod = importlib.import_module(modpath)
    except Exception as e:
        print(f"  [cannot import {modpath}] {e}")
        return
    missing = []
    for n in names:
        try:
            getattr(mod, n)
        except Exception as e:
            missing.append((n, type(e).__name__))
    print(f"  {modpath}: {len(names)} in __all__, {len(missing)} missing at runtime")
    for n, t in missing[:15]:
        print(f"      MISSING {n} ({t})")

print("\n== drivers/dm/__init__ ==")
check_all("ao_shaping.drivers.dm",
          ["HadamardDM","MicroDM","MicroDMConnectionError","MicroDMError","MicroDMParams","MicroDMVoltageError","ZernikeDM"])
print("\n== drivers/dm/micro/__init__ (sample) ==")
check_all("ao_shaping.drivers.dm.micro",
          ["ARRAY_SIDE","AsyncMicroDM","AsyncR50Controller","CHANNELS_PER_CONTROLLER","DM_NUM","Group","Header","MicroDM","MicroDMAsync","R50Controller","VoltageConverter","WiringMap","voltages_to_payload"])
print("\n== drivers/dm/nlight/__init__ (sample) ==")
check_all("ao_shaping.drivers.dm.nlight", ["DISABLED_ACTUATORS","DM_NUM"])
