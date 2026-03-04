import pandas as pd
import matplotlib.pyplot as plt
import pytest

from ao_shaping.wf.DM_wfs import optimizer_rms


def test_wf_optimizer():
    """Test wavefront optimizer - requires DM and WFS hardware"""
    pytest.skip("Requires DM and WFS hardware")
