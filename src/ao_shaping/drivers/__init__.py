from .wfs.thorlab_wfs import WFSManager as Thorlab_WFS
from .wfs.thorlab_wfs import MlaRes
from .dm.NLight import NLight as NlightDM

__all__ = ["Thorlab_WFS", "MlaRes"]
__all__ += ["NlightDM"]

# Try to import CameraStreamManager, but make it optional
try:
    from .ccd.daheng import CameraStreamManager
    __all__ += ["CameraStreamManager"]
except (ImportError, NameError) as e:
    import logging
    logging.getLogger(__name__).warning(f"CameraStreamManager not available: {e}")
    CameraStreamManager = None
