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

# Try to import SantecSLM200, but make it optional
try:
    from .slm.santec_slm200 import SantecSLM200, SantecSLM200Error
    __all__ += ["SantecSLM200", "SantecSLM200Error"]
except ImportError as e:
    import logging
    logging.getLogger(__name__).warning(f"SantecSLM200 not available: {e}")
    SantecSLM200 = None
    SantecSLM200Error = None

# Try to import PyVISA base components, but make them optional
try:
    from .visa_base import (
        VisaResourceManager,
        VisaInstrument,
        VisaInstrumentFactory,
        VisaError,
        is_pyvisa_available,
        list_visa_resources,
        open_visa_instrument,
    )
    __all__ += [
        "VisaResourceManager",
        "VisaInstrument",
        "VisaInstrumentFactory",
        "VisaError",
        "is_pyvisa_available",
        "list_visa_resources",
        "open_visa_instrument",
    ]
except ImportError as e:
    import logging
    logging.getLogger(__name__).debug(f"PyVISA components not available: {e}")
    VisaResourceManager = None
    VisaInstrument = None
    VisaInstrumentFactory = None
    VisaError = None
    is_pyvisa_available = lambda: False
    list_visa_resources = None
    open_visa_instrument = None

# Try to import SLM VISA wrapper, but make it optional
try:
    from .slm.santec_slm200_visa import SantecSLM200Visa, create_slm_visa_instrument
    __all__ += ["SantecSLM200Visa", "create_slm_visa_instrument"]
except ImportError as e:
    import logging
    logging.getLogger(__name__).debug(f"SantecSLM200Visa not available: {e}")
    SantecSLM200Visa = None
    create_slm_visa_instrument = None
