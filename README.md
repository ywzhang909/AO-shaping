# AO-Shaping

AO-Shaping is an adaptive optics (AO) system control and optimization framework designed for wavefront correction and beam shaping applications. The project provides implementations of various optimization algorithms for controlling deformable mirrors (DM) and wavefront sensors (WFS) to achieve precise optical corrections.

## Features

- **Adaptive Optics Control**: Real-time wavefront correction using deformable mirrors
- **Multiple Optimization Algorithms**: Implementation of ADAM, reinforcement learning, and other optimization methods
- **Hardware Integration**: Support for Thorlabs WFS, NLight DM, and Daheng cameras
- **Data Processing & Visualization**: Tools for analyzing and visualizing AO system performance
- **Simulation Capabilities**: Simulate DM behavior for testing and development

## Project Structure

```
ao-shaping/
├── src/
│   └── ao_shaping/
│       ├── algorithm/       # Optimization algorithms (ADAM, etc.)
│       ├── drivers/         # Hardware drivers (DM, WFS, CCD)
│       ├── optimizer/       # Optimization implementations
│       ├── display/         # Visualization components
│       ├── utils/           # Utility functions
│       └── base/            # Base classes
├── tests/                  # Unit tests
├── scripts/                # Utility scripts
├── notebooks/              # Jupyter notebooks for analysis
├── data/                   # Data files and experimental results
├── docs/                   # Documentation
└── requirements.txt        # Project dependencies
```

## Installation

### Prerequisites

- Python 3.13+
- Required hardware drivers (if using physical devices)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/your-username/ao-shaping.git
cd ao-shaping
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Main Components

1. **Wavefront Correction Runner** (`wf_runner.py`):
   ```bash
   python src/ao_shaping/wf_runner.py
   ```

2. **Beam Shaping Runner** (`axis_beam_runner.py`):
   ```bash
   python src/ao_shaping/axis_beam_runner.py
   ```

3. **Combined Runner** (`combined_runner.py`):
   ```bash
   python src/ao_shaping/combined_runner.py
   ```

### Configuration

The system can be configured through environment variables:

- `Far_Cam_ID`: Far field camera ID (default: 0)
- `Near_Cam_ID`: Near field camera ID (default: 1)

Set these in the `.env` file or as system environment variables.

## Hardware Support

- **Deformable Mirrors**: NLight DM64
- **Wavefront Sensors**: Thorlabs WFS series
- **Cameras**: Daheng industrial cameras
- **Translation Stages**: Serial port controlled stages

## Development

### Running Tests

```bash
pytest tests/
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This project uses various open-source libraries and frameworks
- Special thanks to the adaptive optics research community for their contributions