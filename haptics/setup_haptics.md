# Haptic Device Setup and Integration

## Overview
This document provides instructions for setting up haptic devices to work with the Adaptive Hologram Content Customization project.

### Supported Haptic Devices
- **VR Controllers**: Compatible with Unity XR for Oculus, HTC Vive, etc.
- **Haptic Gloves**: Use SDK specific to your device (e.g., Ultraleap, Teslasuit, etc.).

### Setup Instructions
1. **Install Haptic SDK**: Download and integrate the SDK for your haptic device. For VR controllers, install the Unity XR plugin.
2. **Configure Unity Project**:
   - Go to **Edit > Project Settings > XR Plug-in Management**.
   - Enable support for your device (e.g., Oculus, OpenVR).
3. **Testing Haptic Signals**:
   - Open the `HapticFeedbackController` scene.
   - Adjust `HapticFeedbackController` parameters to test signal strength and response.

### Troubleshooting
- **No Haptic Response**: Verify that the device is connected and recognized by Unity XR or the relevant SDK.
- **Inconsistent Feedback**: Adjust intensity values in `HapticFeedbackController.cs` or `HapticInteractionHandler.cs`.

