"""Demo script for mock devices.

This script demonstrates how to use the mock devices for testing
the digital twin device management system without actual hardware.
"""

from loguru import logger

from ao_shaping.drivers import (
    DeviceRegistry,
    DeviceType,
    MockCamera,
    MockDM,
    MockFilter,
    MockLaser,
    MockSLM,
    MockStage,
    MockWFS,
    get_global_registry,
)


def demo_registry():
    """Demonstrate device registry with mock devices."""
    logger.info("=" * 60)
    logger.info("Device Registry Demo")
    logger.info("=" * 60)

    # Create a registry
    registry = DeviceRegistry()

    # Create and register mock devices
    camera = MockCamera(device_id="cam_01", resolution=(1024, 1024))
    slm = MockSLM(device_id="slm_01", resolution=(1920, 1080))
    dm = MockDM(device_id="dm_01", n_actuators=64)
    wfs = MockWFS(device_id="wfs_01", n_lenslets=32)
    stage = MockStage(device_id="stage_x", axis="X")
    laser = MockLaser(device_id="laser_01")
    filter_wheel = MockFilter(device_id="filter_01")

    # Register all devices with aliases and tags
    registry.register(camera, alias="main_camera", tags=["imaging", "primary"])
    registry.register(slm, alias="phase_modulator", tags=["modulation"])
    registry.register(dm, alias="deformable_mirror", tags=["wavefront_control"])
    registry.register(wfs, alias="wavefront_sensor", tags=["wavefront_control", "measurement"])
    registry.register(stage, alias="x_stage", tags=["positioning"])
    registry.register(laser, alias="source", tags=["illumination"])
    registry.register(filter_wheel, alias="filter", tags=["optics"])

    logger.info(f"Registered {len(registry)} devices")
    logger.info(f"Aliases: {registry.list_aliases()}")

    # Find devices by type
    cameras = registry.find_by_type(DeviceType.CAMERA)
    logger.info(f"Found {len(cameras)} camera(s)")

    # Find devices by tag
    wavefront_devices = registry.find_by_tag("wavefront_control")
    logger.info(f"Found {len(wavefront_devices)} wavefront control device(s)")

    return registry


def demo_camera():
    """Demonstrate mock camera."""
    logger.info("\n" + "=" * 60)
    logger.info("Mock Camera Demo")
    logger.info("=" * 60)

    camera = MockCamera(device_id="demo_cam", resolution=(512, 512))

    with camera:
        # Capture single image
        img = camera.capture()
        logger.info(f"Captured image: shape={img.shape}, dtype={img.dtype}")
        logger.info(f"Image stats: min={img.min()}, max={img.max()}, mean={img.mean():.2f}")

        # Capture averaged image
        img_avg = camera.capture(n_samples=5)
        logger.info(f"Averaged image: shape={img_avg.shape}")

        # Adjust parameters
        camera.set_parameter_value("exposure_time_ms", 50.0)
        camera.set_parameter_value("gain", 2.0)
        logger.info(f"Parameters: {camera.list_parameters()}")

        # Get hardware info
        info = camera.get_hardware_info()
        logger.info(f"Hardware info: {info}")

        # Get twin state
        twin_state = camera.get_twin_state()
        logger.info(f"Twin state keys: {list(twin_state.keys())}")


def demo_slm():
    """Demonstrate mock SLM."""
    logger.info("\n" + "=" * 60)
    logger.info("Mock SLM Demo")
    logger.info("=" * 60)

    import numpy as np

    slm = MockSLM(device_id="demo_slm", resolution=(800, 600))

    with slm:
        # Create a phase pattern matching SLM resolution (width, height)
        resolution = slm.get_resolution()
        # meshgrid returns (height, width) shape, transpose to match (width, height)
        x = np.linspace(0, 2 * np.pi, resolution[0])
        y = np.linspace(0, 2 * np.pi, resolution[1])
        xx, yy = np.meshgrid(x, y)
        phase_pattern = np.sin(xx) * np.cos(yy) * np.pi
        phase_pattern = phase_pattern.T  # Transpose to match (width, height)

        # Write phase pattern
        slm.write_phase(phase_pattern)
        logger.info("Phase pattern written to SLM")

        # Get current pattern
        current = slm.get_current_pattern()
        logger.info(f"Current pattern: shape={current.shape}, dtype={current.dtype}")

        # Get twin state
        twin_state = slm.get_twin_state()
        logger.info(f"SLM twin state: frame_count={twin_state['hardware']['frame_count']}")


def demo_dm():
    """Demonstrate mock deformable mirror."""
    logger.info("\n" + "=" * 60)
    logger.info("Mock DM Demo")
    logger.info("=" * 60)

    import numpy as np

    dm = MockDM(device_id="demo_dm", n_actuators=64)

    with dm:
        # Apply some voltages (create a defocus pattern)
        voltages = np.zeros(64)
        # Simple defocus pattern: higher voltages at edges
        for i in range(8):
            for j in range(8):
                idx = i * 8 + j
                r2 = (i - 3.5) ** 2 + (j - 3.5) ** 2
                voltages[idx] = r2 * 10  # Higher voltage at edges

        dm.apply_voltages(voltages)
        logger.info("Voltages applied to DM")

        # Get surface shape
        surface = dm.get_surface()
        logger.info(f"Surface shape: {surface.shape}")
        logger.info(f"Surface range: [{surface.min():.2f}, {surface.max():.2f}] nm")

        # Reset DM
        dm.reset()
        logger.info("DM reset")


def demo_wfs():
    """Demonstrate mock wavefront sensor."""
    logger.info("\n" + "=" * 60)
    logger.info("Mock WFS Demo")
    logger.info("=" * 60)

    wfs = MockWFS(device_id="demo_wfs", n_lenslets=16)

    with wfs:
        # Measure wavefront
        wavefront = wfs.measure_wavefront()
        logger.info(f"Measured wavefront: shape={wavefront.shape}")
        logger.info(f"Wavefront PV: {wavefront.max() - wavefront.min():.3f} rad")

        # Get spot image
        spots = wfs.get_spot_image()
        logger.info(f"Spot image: shape={spots.shape}")

        # Fit Zernike
        zernike_coeffs = wfs.fit_zernike(wavefront, n_modes=15)
        logger.info(f"Zernike coefficients: {len(zernike_coeffs)} modes")


def demo_stage():
    """Demonstrate mock motion stage."""
    logger.info("\n" + "=" * 60)
    logger.info("Mock Stage Demo")
    logger.info("=" * 60)

    stage = MockStage(device_id="demo_stage", axis="Z", travel_range=(0, 50))

    with stage:
        logger.info(f"Initial position: {stage.get_position():.3f} mm")

        # Move to position
        stage.move_to(25.0)
        logger.info(f"After move: {stage.get_position():.3f} mm")

        # Relative move
        stage.move_relative(5.0)
        logger.info(f"After relative move: {stage.get_position():.3f} mm")

        # Home
        stage.home()
        logger.info(f"After home: {stage.get_position():.3f} mm")


def demo_laser():
    """Demonstrate mock laser."""
    logger.info("\n" + "=" * 60)
    logger.info("Mock Laser Demo")
    logger.info("=" * 60)

    laser = MockLaser(device_id="demo_laser")

    with laser:
        # Set parameters
        laser.set_wavelength(633.0)
        laser.set_power(50.0)
        logger.info(f"Wavelength: {laser.get_wavelength():.1f} nm")
        logger.info(f"Power setting: {laser.get_power():.1f} mW")

        # Enable output
        laser.enable_output(True)
        logger.info(f"Output enabled: {laser.is_output_enabled()}")
        logger.info(f"Actual output power: {laser.get_power():.1f} mW")

        # Disable output
        laser.enable_output(False)
        logger.info(f"After disable: {laser.get_power():.1f} mW")


def demo_filter():
    """Demonstrate mock filter wheel."""
    logger.info("\n" + "=" * 60)
    logger.info("Mock Filter Demo")
    logger.info("=" * 60)

    filters = ["Open", "ND1", "ND2", "Red-633", "Green-532", "Blue-488", "Block"]
    fw = MockFilter(device_id="demo_filter", filters=filters)

    with fw:
        logger.info(f"Available filters: {fw.get_filter_list()}")
        logger.info(f"Current filter: {fw.get_current_filter()}")

        # Move by position
        fw.move_to_position(3)
        logger.info(f"After move to pos 3: {fw.get_current_filter()}")

        # Move by name
        fw.move_to_filter("ND1")
        logger.info(f"After move to ND1: position={fw.get_current_position()}")


def demo_global_registry():
    """Demonstrate global registry usage."""
    logger.info("\n" + "=" * 60)
    logger.info("Global Registry Demo")
    logger.info("=" * 60)

    # Get global registry
    registry = get_global_registry()

    # Add a device to global registry
    camera = MockCamera(device_id="global_cam")
    registry.register(camera, alias="global_test_camera")

    # Get all twin states
    states = registry.get_all_twin_states()
    logger.info(f"Global registry has {len(states)} device(s) with twin state")

    # Get snapshot
    snapshot = registry.get_twin_snapshot()
    logger.info(f"Snapshot: {len(snapshot['devices'])} device(s)")
    logger.info(f"Device types: {snapshot['registry_info']['device_types']}")


def main():
    """Run all demos."""
    logger.add(
        lambda msg: print(msg, end=""),
        level="INFO",
        format="<level>{message}</level>",
    )

    logger.info("\n" + "=" * 60)
    logger.info("Mock Devices Demo")
    logger.info("=" * 60)

    try:
        # Run individual device demos
        demo_camera()
        demo_slm()
        demo_dm()
        demo_wfs()
        demo_stage()
        demo_laser()
        demo_filter()

        # Run registry demos
        registry = demo_registry()
        demo_global_registry()

        # Batch operations on registry
        logger.info("\n" + "=" * 60)
        logger.info("Batch Operations")
        logger.info("=" * 60)

        # Health check all
        health = registry.health_check_all()
        for device_id, (is_healthy, msg) in health.items():
            status = "healthy" if is_healthy else "unhealthy"
            logger.info(f"  {device_id[:8]}: {status} ({msg})")

        # Get all status
        all_status = registry.get_all_status()
        logger.info(f"Collected status for {len(all_status)} device(s)")

        logger.info("\n" + "=" * 60)
        logger.info("Demo completed successfully!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    main()
