"""
Tests for Zernike Response Matrix UI - shift_x and shift_y parameters (Task 1)

These tests verify that:
1. Session state variables zrm_shift_x and zrm_shift_y are initialized to 0
2. SLM Settings section contains shift_x and shift_y number inputs
3. ZernikeSLM constructor receives shift_x and shift_y parameters
"""

import streamlit as st
import pytest

from unittest.mock import patch, MagicMock, PropertyMock
import pytest
import sys
from pathlib import Path
from datetime import datetime

import ast

_UI_SOURCE = (
    Path(__file__).resolve().parents[4]
    / "src" / "ao_shaping" / "gui" / "zernike" / "zernike_response_matrix_ui.py"
)
_UI_TREE = ast.parse(_UI_SOURCE.read_text(encoding="utf-8-sig"))
_WIDGET_CALLS = ("number_input", "checkbox", "selectbox", "text_input", "slider", "toggle")


def _widget_call(key: str):
    """The Streamlit widget call whose value lands in ``st.session_state.<key>``.

    Asserted on the parsed UI tree rather than on source text: a string check keeps
    passing after the widget is deleted, as long as the label survives in a comment.
    Covers the two-hop form too (``new_exp = st.number_input(...)`` and only then
    ``st.session_state.zrm_wfs_exp_time = new_exp``).
    """
    hops: set[str] = set()
    for node in ast.walk(_UI_TREE):
        if not isinstance(node, ast.Assign):
            continue
        bound = any(
            isinstance(t, ast.Attribute)
            and t.attr == key
            and isinstance(t.value, ast.Attribute)
            and t.value.attr == "session_state"
            for t in node.targets
        )
        if not bound:
            continue
        value = node.value
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and value.func.attr in _WIDGET_CALLS
        ):
            return value
        if isinstance(value, ast.Name):
            hops.add(value.id)
    for node in ast.walk(_UI_TREE):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        func = node.value.func
        if not isinstance(func, ast.Attribute) or func.attr not in _WIDGET_CALLS:
            continue
        if any(isinstance(t, ast.Name) and t.id in hops for t in node.targets):
            return node.value
    return None


def _label_of(call) -> str:
    """First positional argument of a widget call; '' when the label is dynamic."""
    if call is not None and call.args and isinstance(call.args[0], ast.Constant):
        return str(call.args[0].value)
    return ""


def _call_names(attr: str) -> list:
    return [
        node
        for node in ast.walk(_UI_TREE)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == attr
    ]


# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _function(name: str):
    """A FunctionDef from the UI module, so assertions bind to the right scope."""
    return next(
        (n for n in ast.walk(_UI_TREE)
         if isinstance(n, ast.FunctionDef) and n.name == name),
        None,
    )


class TestSLMShiftParameters:
    """Test suite for shift_x and shift_y parameters in Zernike Response Matrix UI"""

    def _create_mock_session_state(self, **initial_values):
        """Create a mock session state that behaves like st.session_state"""
        class MockSessionState:
            def __init__(self, data=None):
                self._data = data or {}

            def __getattr__(self, name):
                if name.startswith('_'):
                    # Let Python handle private attributes normally
                    raise AttributeError(f"'MockSessionState' object has no attribute '{name}'")
                return self._data.get(name)

            def __setattr__(self, name, value):
                if name.startswith('_'):
                    super().__setattr__(name, value)
                else:
                    if not hasattr(self, '_data'):
                        super().__setattr__('_data', {})
                    self._data[name] = value

            def __contains__(self, key):
                return key in self._data

            def __getitem__(self, key):
                return self._data[key]

            def __setitem__(self, key, value):
                self._data[key] = value

            def get(self, key, default=None):
                return self._data.get(key, default)

        mock = MockSessionState()
        for key, value in initial_values.items():
            mock._data[key] = value
        return mock

    def test_session_state_initialization(self):
        """Test that zrm_shift_x and zrm_shift_y are initialized to 0"""
        mock_state = self._create_mock_session_state()

        with patch.object(st, 'session_state', mock_state):
            from ao_shaping.gui.zernike.zernike_response_matrix_ui import _initialize_state

            _initialize_state()

            assert mock_state.zrm_shift_x == 0, "zrm_shift_x should default to 0"
            assert mock_state.zrm_shift_y == 0, "zrm_shift_y should default to 0"

    def test_shift_inputs_in_slm_settings(self):
        """shift_x / shift_y must be real widgets bound to their session keys."""
        for key, label in (("zrm_shift_x", "SLM Shift X (像素)"),
                           ("zrm_shift_y", "SLM Shift Y (像素)")):
            call = _widget_call(key)
            assert call is not None, f"{key} must be bound by a Streamlit widget"
            assert call.func.attr == "number_input", f"{key} should be an st.number_input"
            assert _label_of(call) == label, (
                f"{key} label should be {label!r}, got {_label_of(call)!r}"
            )

    def test_zernikeslm_receives_shift_params(self):
        """Test that ZernikeSLM constructor receives shift_x and shift_y"""
        mock_zernikeslm = MagicMock()
        mock_instance = MagicMock()
        mock_zernikeslm.return_value = mock_instance

        mock_state = self._create_mock_session_state(
            zrm_slm_selection=1,
            zrm_slm_wavelength=532,
            zrm_slm_n_max=10,
            zrm_shift_x=50,
            zrm_shift_y=-30,
            zrm_slm=None,
            zrm_slm_connected=False,
        )

        with patch.object(st, 'session_state', mock_state):
            with patch("ao_shaping.gui.zernike.zernike_response_matrix_ui.ZernikeSLM", mock_zernikeslm):
                from ao_shaping.gui.zernike.zernike_response_matrix_ui import connect_slm

                connect_slm()

                mock_zernikeslm.assert_called_once()
                call_args = mock_zernikeslm.call_args

                assert call_args.kwargs.get("shift_x") == 50, "shift_x should be 50"
                assert call_args.kwargs.get("shift_y") == -30, "shift_y should be -30"
                assert call_args.kwargs.get("slm_number") == 1
                assert call_args.kwargs.get("wavelength") == 532
                assert call_args.kwargs.get("n_max") == 10

    def test_shift_defaults_in_initialize_state(self):
        """Test that _initialize_state sets proper defaults for shift parameters"""
        mock_state = self._create_mock_session_state()

        with patch.object(st, 'session_state', mock_state):
            from ao_shaping.gui.zernike.zernike_response_matrix_ui import _initialize_state

            _initialize_state()

            assert mock_state.zrm_shift_x == 0
            assert mock_state.zrm_shift_y == 0

    def test_connect_slm_with_default_shifts(self):
        """Test connect_slm uses default shift values when not specified"""
        mock_zernikeslm = MagicMock()
        mock_instance = MagicMock()
        mock_zernikeslm.return_value = mock_instance

        mock_state = self._create_mock_session_state(
            zrm_slm_selection=1,
            zrm_slm_wavelength=532,
            zrm_slm_n_max=10,
            zrm_shift_x=0,
            zrm_shift_y=0,
            zrm_slm=None,
            zrm_slm_connected=False,
        )

        with patch.object(st, 'session_state', mock_state):
            with patch("ao_shaping.gui.zernike.zernike_response_matrix_ui.ZernikeSLM", mock_zernikeslm):
                from ao_shaping.gui.zernike.zernike_response_matrix_ui import connect_slm

                connect_slm()

                call_args = mock_zernikeslm.call_args
                assert call_args.kwargs.get("shift_x") == 0
                assert call_args.kwargs.get("shift_y") == 0


class TestWFSSettings:
    """Test suite for WFS Settings section in Zernike Response Matrix UI (Task 2)"""

    def _create_mock_session_state(self, **initial_values):
        """Create a mock session state that behaves like st.session_state"""
        class MockSessionState:
            def __init__(self, data=None):
                self._data = data or {}

            def __getattr__(self, name):
                if name.startswith('_'):
                    raise AttributeError(f"'MockSessionState' object has no attribute '{name}'")
                return self._data.get(name)

            def __setattr__(self, name, value):
                if name.startswith('_'):
                    super().__setattr__(name, value)
                else:
                    if not hasattr(self, '_data'):
                        super().__setattr__('_data', {})
                    self._data[name] = value

            def __contains__(self, key):
                return key in self._data

            def __getitem__(self, key):
                return self._data[key]

            def __setitem__(self, key, value):
                self._data[key] = value

            def get(self, key, default=None):
                return self._data.get(key, default)

        mock = MockSessionState()
        for key, value in initial_values.items():
            mock._data[key] = value
        return mock

    def test_wfs_session_state_initialization(self):
        """Test that WFS session state variables are initialized"""
        mock_state = self._create_mock_session_state()

        with patch.object(st, 'session_state', mock_state):
            from ao_shaping.gui.zernike.zernike_response_matrix_ui import _initialize_state

            _initialize_state()

            assert mock_state.zrm_wfs_mla_res == "768", "zrm_wfs_mla_res should default to '768'"
            assert mock_state.zrm_wfs_exp_time == 0.01, "zrm_wfs_exp_time should default to 0.01"
            assert mock_state.zrm_wfs_pupil_diameter == 2.0, "zrm_wfs_pupil_diameter should default to 2.0"
            assert mock_state.zrm_wfs_pupil_center_x == 0.0, "zrm_wfs_pupil_center_x should default to 0.0"
            assert mock_state.zrm_wfs_pupil_center_y == 0.0, "zrm_wfs_pupil_center_y should default to 0.0"

    def test_wfs_settings_section_exists(self):
        """The WFS section needs a real subheader and widget-driven keys."""
        headers = {_label_of(node) for node in _call_names("subheader")}
        assert "WFS参数" in headers, "the WFS section needs st.subheader('WFS参数')"
        for key in ("zrm_wfs_mla_res", "zrm_wfs_exp_time", "zrm_wfs_pupil_diameter",
                    "zrm_wfs_pupil_center_x", "zrm_wfs_pupil_center_y"):
            assert _widget_call(key) is not None, f"{key} should be driven by a widget"

    def test_wfsmanager_receives_parameters(self):
        """Test that WFSManager receives WFS parameters"""
        mock_wfsmanager = MagicMock()
        mock_instance = MagicMock()
        mock_instance.get_exposure_time_range.return_value = (0.0, 0.086, 0)
        mock_wfsmanager.return_value = mock_instance

        mock_state = self._create_mock_session_state(
            zrm_wfs_mla_res="768",
            zrm_wfs_exp_time=10.0,
            zrm_wfs_pupil_diameter=3.5,
            zrm_wfs_pupil_center_x=1.5,
            zrm_wfs_pupil_center_y=-0.5,
            zrm_wfs=None,
            zrm_wfs_connected=False,
        )

        with patch.object(st, 'session_state', mock_state):
            with patch("ao_shaping.gui.zernike.zernike_response_matrix_ui.ThorlabWFS", mock_wfsmanager):
                from ao_shaping.gui.zernike.zernike_response_matrix_ui import connect_wfs

                connect_wfs()

                mock_wfsmanager.assert_called_once()
                call_args = mock_wfsmanager.call_args

                # Check that WFSManager was called with correct parameters
                assert call_args.kwargs.get("exposure_time") == 10.0, "exposure_time should be 10.0"
                assert call_args.kwargs.get("pupil_diameter") == 3.5, "pupil_diameter should be 3.5"
                assert call_args.kwargs.get("pupil_center") == (1.5, -0.5), "pupil_center should be (1.5, -0.5)"

    def test_mla_res_options(self):
        """Five resolutions on offer, each displayed as WxH through format_func."""
        call = _widget_call("zrm_wfs_mla_res")
        assert call is not None and call.func.attr == "selectbox", (
            "zrm_wfs_mla_res should be an st.selectbox"
        )
        options = next((k for k in call.keywords if k.arg == "options"), None)
        assert options is not None and len(options.value.elts) == 5, (
            "the selectbox must offer exactly 5 MLA resolutions"
        )
        fmt = next((k for k in call.keywords if k.arg == "format_func"), None)
        assert fmt is not None, "resolutions are shown through format_func"
        shown = {n.value for n in ast.walk(fmt.value)
                 if isinstance(n, ast.Constant) and isinstance(n.value, str)}
        assert {"1280x1024", "1024x1024", "768x768", "512x512", "320x320"} <= shown, (
            f"every option must map to a WxH label, got {sorted(shown)}"
        )


class TestCalibrationParameters:
    """Test suite for excluded_piston, compute_inverses, and verbose parameters (Task 3)"""

    def _create_mock_session_state(self, **initial_values):
        """Create a mock session state that behaves like st.session_state"""
        class MockSessionState:
            def __init__(self, data=None):
                self._data = data or {}

            def __getattr__(self, name):
                if name.startswith('_'):
                    raise AttributeError(f"'MockSessionState' object has no attribute '{name}'")
                return self._data.get(name)

            def __setattr__(self, name, value):
                if name.startswith('_'):
                    super().__setattr__(name, value)
                else:
                    if not hasattr(self, '_data'):
                        super().__setattr__('_data', {})
                    self._data[name] = value

            def __contains__(self, key):
                return key in self._data

            def __getitem__(self, key):
                return self._data[key]

            def __setitem__(self, key, value):
                self._data[key] = value

            def get(self, key, default=None):
                return self._data.get(key, default)

        mock = MockSessionState()
        for key, value in initial_values.items():
            mock._data[key] = value
        return mock

    def test_session_state_excluded_piston(self):
        """Test that zrm_excluded_piston is initialized to True"""
        mock_state = self._create_mock_session_state()

        with patch.object(st, 'session_state', mock_state):
            from ao_shaping.gui.zernike.zernike_response_matrix_ui import _initialize_state

            _initialize_state()

            assert mock_state.zrm_excluded_piston is True, "zrm_excluded_piston should default to True"

    def test_session_state_compute_inverses(self):
        """Test that zrm_compute_inverses is initialized to True"""
        mock_state = self._create_mock_session_state()

        with patch.object(st, 'session_state', mock_state):
            from ao_shaping.gui.zernike.zernike_response_matrix_ui import _initialize_state

            _initialize_state()

            assert mock_state.zrm_compute_inverses is True, "zrm_compute_inverses should default to True"

    def test_session_state_verbose(self):
        """Test that zrm_verbose is initialized to True (optional)"""
        mock_state = self._create_mock_session_state()

        with patch.object(st, 'session_state', mock_state):
            from ao_shaping.gui.zernike.zernike_response_matrix_ui import _initialize_state

            _initialize_state()

            assert mock_state.zrm_verbose is True, "zrm_verbose should default to True"

    def test_excluded_piston_checkbox_exists(self):
        """The piston flag must be a checkbox wired to its session key."""
        call = _widget_call("zrm_excluded_piston")
        assert call is not None and call.func.attr == "checkbox", (
            "zrm_excluded_piston should be an st.checkbox"
        )
        assert "Piston" in _label_of(call), (
            f"label should name Piston, got {_label_of(call)!r}"
        )

    def test_compute_inverses_checkbox_exists(self):
        """The inverse-matrix flag must be a checkbox wired to its session key."""
        call = _widget_call("zrm_compute_inverses")
        assert call is not None and call.func.attr == "checkbox", (
            "zrm_compute_inverses should be an st.checkbox"
        )

    def test_verbose_checkbox_exists(self):
        """The verbose flag must be a checkbox wired to its session key."""
        call = _widget_call("zrm_verbose")
        assert call is not None and call.func.attr == "checkbox", (
            "zrm_verbose should be an st.checkbox"
        )

    def test_calibrate_receives_parameters(self):
        """Test that calibrate_zernike_response_matrix receives excluded_piston, compute_inverses, verbose"""
        mock_calibrate = MagicMock()
        mock_result = MagicMock()
        mock_calibrate.return_value = mock_result

        mock_state = self._create_mock_session_state(
            zrm_slm_connected=True,
            zrm_wfs_connected=True,
            zrm_n_max=10,
            zrm_magnitude=0.1,
            zrm_n_cycles=2,
            zrm_n_averages=5,
            zrm_wait_time=0.1,
            zrm_excluded_piston=True,
            zrm_compute_inverses=True,
            zrm_verbose=True,
            zrm_calibration_running=False,
            zrm_storage_dir=str(Path(__file__).resolve().parents[4] / "data" / "zernike_calibration"),
        )

        # Mock the SLM and WFS objects
        mock_slm = MagicMock()
        mock_wfs = MagicMock()
        mock_state.zrm_slm = mock_slm
        mock_state.zrm_wfs = mock_wfs

        with patch.object(st, 'session_state', mock_state):
            with patch("ao_shaping.gui.zernike.zernike_response_matrix_ui.calibrate_zernike_response_matrix", mock_calibrate):
                from ao_shaping.gui.zernike.zernike_response_matrix_ui import render_calibrate_mode

                # Call the function (it checks for button click, we need to mock st.button)
                with patch.object(st, 'button', return_value=True):
                    with patch.object(st, 'text_input', return_value="test_file"):
                        with patch("ao_shaping.gui.zernike.zernike_response_matrix_ui.save_zernike_response_matrix"):
                            with patch.object(st, 'success'):
                                with patch.object(st, 'error'):
                                    with patch.object(st, 'warning'):
                                        with patch.object(st, 'info'):
                                            with patch.object(st, 'metric'):
                                                with patch.object(st, 'divider'):
                                                    with patch("matplotlib.pyplot.subplots"):
                                                        with patch.object(st, 'pyplot'):
                                                            render_calibrate_mode()

                # Check that calibrate_zernike_response_matrix was called with correct parameters
                mock_calibrate.assert_called_once()
                call_args = mock_calibrate.call_args
                assert call_args.kwargs.get("excluded_piston") is True, "excluded_piston should be True"
                assert call_args.kwargs.get("compute_inverses") is True, "compute_inverses should be True"
                assert call_args.kwargs.get("verbose") is True, "verbose should be True"


class TestProgressVisualization:
    """Test suite for progress bar and dynamic visualization (Task 5)"""

    def _create_mock_session_state(self, **initial_values):
        """Create a mock session state that behaves like st.session_state"""
        class MockSessionState:
            def __init__(self, data=None):
                self._data = data or {}

            def __getattr__(self, name):
                if name.startswith('_'):
                    raise AttributeError(f"'MockSessionState' object has no attribute '{name}'")
                return self._data.get(name)

            def __setattr__(self, name, value):
                if name.startswith('_'):
                    super().__setattr__(name, value)
                else:
                    if not hasattr(self, '_data'):
                        super().__setattr__('_data', {})
                    self._data[name] = value

            def __contains__(self, key):
                return key in self._data

            def __getitem__(self, key):
                return self._data[key]

            def __setitem__(self, key, value):
                self._data[key] = value

            def get(self, key, default=None):
                return self._data.get(key, default)

        mock = MockSessionState()
        for key, value in initial_values.items():
            mock._data[key] = value
        return mock

    def test_progress_session_state_initialization(self):
        """Test that progress-related session state is initialized"""
        mock_state = self._create_mock_session_state()

        with patch.object(st, 'session_state', mock_state):
            from ao_shaping.gui.zernike.zernike_response_matrix_ui import _initialize_state

            _initialize_state()

            # Check that progress-related session state variables exist
            assert hasattr(mock_state, 'zrm_progress_file'), "zrm_progress_file should be initialized"
            assert hasattr(mock_state, 'zrm_calibration_running'), "zrm_calibration_running should be initialized"
            assert mock_state.zrm_calibration_running is False, "zrm_calibration_running should default to False"

    def test_progress_bar_placeholder_exists(self):
        """Progress UI must create its placeholders through st.empty()."""
        names = {
            t.id
            for node in ast.walk(_UI_TREE)
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "empty"
            for t in node.targets if isinstance(t, ast.Name)
        }
        assert {"progress_bar", "status_text"} <= names, (
            f"progress placeholders need st.empty(); got {sorted(names)}"
        )

    def test_callback_creates_json_file(self):
        """Test that the callback writes progress to JSON file"""
        import tempfile
        import json

        # Create a temporary progress file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            progress_file = f.name

        try:
            mock_state = self._create_mock_session_state(zrm_progress_file=progress_file)

            with patch.object(st, 'session_state', mock_state):
                from ao_shaping.gui.zernike.zernike_response_matrix_ui import _progress_callback

                # Call the callback with test data
                test_data = {
                    "current_mode": 5,
                    "total_modes": 10,
                    "mode_name": "Tip",
                    "percent": 50.0,
                    "message": "Calibrating mode 5/10: Tip",
                    "timestamp": "2026-04-28T10:00:00",
                }

                _progress_callback(progress_file, test_data)

                # Verify JSON file was created with correct data
                with open(progress_file, "r") as f:
                    saved_data = json.load(f)

                assert saved_data["current_mode"] == 5, "current_mode should be 5"
                assert saved_data["total_modes"] == 10, "total_modes should be 10"
                assert saved_data["percent"] == 50.0, "percent should be 50.0"
                assert saved_data["mode_name"] == "Tip", "mode_name should be 'Tip'"

        finally:
            # Clean up
            if Path(progress_file).exists():
                Path(progress_file).unlink()

    def test_polling_displays_updates(self):
        """The polling fragment must read the progress JSON and move the bar.

        Scoped to the function that has to do it. The previous version read the
        whole module as text, so ``zrm_progress_file`` anywhere in the file
        satisfied it -- ``_progress_callback`` and the session-state init both
        mention it -- while ~45 lines of mock scaffolding ended in ``pass``,
        exercising nothing. JSON decoding is matched as load/loads so a
        refactor of the spelling stays green.
        """
        fn = _function("_render_calibration_polling")
        assert fn is not None, "the polling fragment _render_calibration_polling must exist"

        session_reads = {
            n.args[0].value
            for n in ast.walk(fn)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr == "get" and n.args and isinstance(n.args[0], ast.Constant)
        }
        assert "zrm_progress_file" in session_reads, (
            "polling must read the progress-file path out of session state"
        )

        decoded = [
            n for n in ast.walk(fn)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr in {"load", "loads"}
        ]
        assert decoded, "polling must decode the progress JSON"

        moved = [
            n for n in ast.walk(fn)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr == "progress"
        ]
        assert moved, "polling must update the progress bar"

    def test_threading_for_calibration(self):
        """Test that calibration runs in background thread"""
        mock_thread = MagicMock()
        mock_thread_instance = MagicMock()
        mock_thread.return_value = mock_thread_instance

        mock_state = self._create_mock_session_state(
            zrm_slm_connected=True,
            zrm_wfs_connected=True,
            zrm_n_max=10,
            zrm_magnitude=0.1,
            zrm_n_cycles=2,
            zrm_n_averages=5,
            zrm_wait_time=0.1,
            zrm_excluded_piston=True,
            zrm_compute_inverses=True,
            zrm_verbose=True,
            zrm_calibration_running=False,
            zrm_storage_dir=str(Path(__file__).resolve().parents[4] / "data" / "zernike_calibration"),
            zrm_progress_file="test_progress.json",
        )

        # Mock the SLM and WFS objects
        mock_slm = MagicMock()
        mock_wfs = MagicMock()
        mock_state.zrm_slm = mock_slm
        mock_state.zrm_wfs = mock_wfs

        with patch.object(st, 'session_state', mock_state):
            with patch("ao_shaping.gui.zernike.zernike_response_matrix_ui.threading") as mock_threading:
                mock_threading.Thread.return_value = mock_thread_instance

                from ao_shaping.gui.zernike.zernike_response_matrix_ui import render_calibrate_mode

                # Mock UI elements
                with patch.object(st, 'button', return_value=True):
                    with patch.object(st, 'text_input', return_value="test_file"):
                        with patch.object(st, 'success'):
                            with patch.object(st, 'error'):
                                with patch.object(st, 'warning'):
                                    with patch.object(st, 'info'):
                                        with patch.object(st, 'metric'):
                                            with patch.object(st, 'divider'):
                                                with patch.object(st, 'rerun'):
                                                    with patch("pathlib.Path.mkdir"):
                                                        with patch("pathlib.Path.exists", return_value=False):
                                                            render_calibrate_mode()

                # Verify that Thread was called with correct target
                mock_threading.Thread.assert_called_once()
                call_args = mock_threading.Thread.call_args
                assert call_args.kwargs.get("target") is not None, "Thread should have a target function"
                assert call_args.kwargs.get("daemon") is True, "Thread should be daemon=True"
                assert "args" in call_args.kwargs, "Thread should have args"

    def test_progress_callback_format(self):
        """Test that progress callback writes correct JSON format"""
        import tempfile
        import json

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            progress_file = f.name

        try:
            mock_state = self._create_mock_session_state(zrm_progress_file=progress_file)

            with patch.object(st, 'session_state', mock_state):
                from ao_shaping.gui.zernike.zernike_response_matrix_ui import _progress_callback

                # Test with completion status
                completion_data = {
                    "current_mode": 10,
                    "total_modes": 10,
                    "percent": 100.0,
                    "message": "Calibration complete!",
                    "timestamp": datetime.now().isoformat(),
                    "status": "complete",
                    "result_path": "/path/to/results",
                }

                _progress_callback(progress_file, completion_data)

                with open(progress_file, "r") as f:
                    saved_data = json.load(f)

                assert saved_data["status"] == "complete", "status should be 'complete'"
                assert saved_data["percent"] == 100.0, "percent should be 100.0"
                assert "result_path" in saved_data, "Should include result_path"

                # Test with error status
                error_data = {
                    "percent": -1,
                    "message": "Calibration failed: Connection error",
                    "timestamp": datetime.now().isoformat(),
                    "status": "error",
                }

                _progress_callback(progress_file, error_data)

                with open(progress_file, "r") as f:
                    saved_data = json.load(f)

                assert saved_data["status"] == "error", "status should be 'error'"
                assert saved_data["percent"] == -1, "percent should be -1 for error"

        finally:
            if Path(progress_file).exists():
                Path(progress_file).unlink()
