# ComfyUI A2V Position Adjustment Node

A custom node for ComfyUI that provides advanced image positioning and adjustment capabilities with real-time preview.

## Features

- Real-time preview window for adjusting image position
- Smooth drag & drop functionality
- Scale and rotation controls
- Horizontal and vertical flip options
- Grid overlay for precise positioning
- Status overlay with current parameters
- Keyboard shortcuts for quick adjustments

## Installation

1. Clone this repository into your ComfyUI custom nodes folder:
```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/aiartvn/A2V_Position_Adjustment
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Restart ComfyUI

## Usage

1. Add the "A2V Position Adjustment" node to your workflow
2. Connect an input image and a background image
3. Enable preview to open the adjustment window
4. Use the following controls:
   - Left mouse drag: Move image
   - Mouse wheel: Scale image
   - Mouse wheel (T mode): Rotate image
   - Shift/Ctrl + drag/wheel: Fine control

### Keyboard Shortcuts

- `H`: Toggle horizontal flip
- `V`: Toggle vertical flip
- `G`: Toggle grid overlay
- `R`: Reset all adjustments
- `S`: Toggle scale mode
- `T`: Toggle rotate mode
- `C`: Center image
- `ESC/Enter/Q`: Close preview

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Preview

![Preview](examples/preview.png)

## Example Workflow

An example workflow is provided in the examples folder.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
