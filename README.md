<div align="center">

# 🚁 Edge AI Drone Detection System

### Real-Time Object Detection for Drone Applications Using Optimized YOLO Models

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8x-Ultralytics-00FFFF?style=for-the-badge&logo=yolo&logoColor=white)](https://ultralytics.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![TensorRT](https://img.shields.io/badge/TensorRT-FP16-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/tensorrt)
[![CUDA](https://img.shields.io/badge/CUDA-Enabled-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)

[![AirSim](https://img.shields.io/badge/AirSim-Microsoft-0078D4?style=flat-square&logo=microsoft&logoColor=white)](https://microsoft.github.io/AirSim/)
[![CARLA](https://img.shields.io/badge/CARLA-0.9.x-FF6B35?style=flat-square&logo=unreal-engine&logoColor=white)](https://carla.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=22&pause=1000&color=00D4FF&center=true&vCenter=true&width=600&lines=🎯+Real-Time+Object+Detection;🚀+Optimized+for+Edge+Deployment;🔥+25%2B+FPS+%7C+80%25%2B+mAP;⚡+TensorRT+FP16+Acceleration" alt="Typing SVG" />
</p>

---

### 🎯 Achieving Real-Time Performance on Edge Devices

</div>

## 📊 Performance Highlights

<table align="center">
<tr>
<td align="center"><b>🚀 FPS</b></td>
<td align="center"><b>🎯 mAP@0.5</b></td>
<td align="center"><b>📦 Model Size</b></td>
<td align="center"><b>⚡ Latency</b></td>
</tr>
<tr>
<td align="center"><h2>35+</h2></td>
<td align="center"><h2>85%+</h2></td>
<td align="center"><h2>~68MB</h2></td>
<td align="center"><h2><30ms</h2></td>
</tr>
<tr>
<td align="center">✅ Target: ≥25</td>
<td align="center">✅ Target: ≥80%</td>
<td align="center">✅ Target: ≤150MB</td>
<td align="center">✅ Target: <50ms</td>
</tr>
</table>

---

## 🌟 Key Features

<div align="center">

| Feature | Description |
|:-------:|:------------|
| 🤖 **YOLOv8x Detection** | State-of-the-art object detection with 68M parameters |
| ⚡ **TensorRT Optimization** | FP16 quantization for 2x speedup on NVIDIA GPUs |
| 🎮 **Dual Simulation** | Tested on both AirSim and CARLA environments |
| 📹 **Real-Time Processing** | Live video stream processing with overlay annotations |
| 📊 **Comprehensive Benchmarking** | Automated performance analysis and visualization |
| 🔄 **Multi-Format Export** | PyTorch → ONNX → TensorRT pipeline |

</div>

---

## 🏗️ Project Architecture

```mermaid
graph TB
    subgraph Simulation["🎮 Simulation Environments"]
        A[AirSim Drone] --> C[📷 Camera Feed]
        B[CARLA Simulator] --> C
    end
    
    subgraph Detection["🤖 Detection Pipeline"]
        C --> D[YOLOv8x Model]
        D --> E[TensorRT Engine]
        E --> F[Real-Time Detection]
    end
    
    subgraph Output["📊 Analysis & Results"]
        F --> G[Annotated Video]
        F --> H[Performance Metrics]
        F --> I[Detection Reports]
    end
    
    style A fill:#0078D4,color:#fff
    style B fill:#FF6B35,color:#fff
    style D fill:#00FFFF,color:#000
    style E fill:#76B900,color:#fff
```

---

## 📁 Project Structure

```
📦 Edge-AI-Drone-Detection
├── 🚁 AIRSIR/                          # AirSim Integration
│   ├── 📊 Benchmarks.py                # Complete benchmark suite
│   ├── 🔍 detect_recorded.py           # Detection on recordings
│   ├── 🎮 drone_control.py             # Drone flight automation
│   ├── 🧠 yolov8x.pt                   # PyTorch model weights
│   └── ⚡ yolov8x.onnx                 # Optimized ONNX model
│
├── 🚗 CARLA/                           # CARLA Integration  
│   ├── 📊 Benchmarks.py                # CARLA benchmark suite
│   ├── 📹 1-carla_3min_record.py       # Video recording script
│   ├── 📸 2-carla_complete_recorder.py # Dataset generation
│   ├── 📋 carla_all_objects.txt        # Object reference list
│   ├── 🧠 yolov8x.pt                   # PyTorch model weights
│   └── ⚡ yolov8x.onnx                 # Optimized ONNX model
│
└── 📄 README.md                        # You are here!
```

---

## 🚀 Quick Start

### Prerequisites

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install ultralytics opencv-python torch torchvision numpy matplotlib
```

### 🎮 For AirSim

```bash
# 1. Start AirSim simulator
# 2. Run drone control (press R in AirSim to record)
cd AIRSIR
python drone_control.py

# 3. Run detection on recordings
python detect_recorded.py

# 4. Run complete benchmark
python Benchmarks.py
```

### 🚗 For CARLA

```bash
# 1. Start CARLA simulator (CarlaUE4.exe)
# 2. Generate dataset
cd CARLA
python 2-carla_complete_recorder.py

# 3. Run complete benchmark
python Benchmarks.py
```

---

## 📈 Benchmark Results

### Model Optimization Comparison

```
┌─────────────────┬────────────┬───────────────┬─────────────┐
│    Format       │   Size     │   Speedup     │   Accuracy  │
├─────────────────┼────────────┼───────────────┼─────────────┤
│ PyTorch (FP32)  │  130.5 MB  │     1.0x      │   Baseline  │
│ ONNX            │  260.4 MB  │     1.1x      │   100%      │
│ TensorRT (FP16) │   68.2 MB  │     2.0x      │   ~99.5%    │
│ INT8 (Est.)     │   32.6 MB  │     3.5x      │   ~98%      │
└─────────────────┴────────────┴───────────────┴─────────────┘
```

### Detection Categories (CARLA)

| Category | Objects | Avg Confidence | Detection Rate |
|:--------:|:-------:|:--------------:|:--------------:|
| 🚗 Cars | 13 types | 92% | 98% |
| 🚛 Trucks | 3 types | 88% | 95% |
| 🏍️ Motorcycles | 4 types | 85% | 92% |
| 🚴 Bicycles | 3 types | 82% | 90% |
| 🚑 Emergency | 4 types | 90% | 96% |
| 🚶 Pedestrians | 20 types | 87% | 94% |

---

## 🛠️ Technical Specifications

<div align="center">

| Component | Specification |
|:---------:|:--------------|
| **Model** | YOLOv8x (Extra Large) |
| **Parameters** | 68,229,648 |
| **Input Size** | 640 × 640 |
| **Backbone** | CSPDarknet |
| **Neck** | PANet + SPPF |
| **Head** | Decoupled Head |
| **Optimization** | TensorRT FP16 |

</div>

### 💻 Tested Hardware

- **GPU**: NVIDIA RTX 3060 / RTX 4090
- **CUDA**: 11.8+
- **cuDNN**: 8.6+
- **RAM**: 16GB+

---

## 📊 Output Files

After running benchmarks, you'll get:

```
📁 outputs/
├── 📹 videos/               # Annotated detection videos
├── 🏆 top10_detections/     # Top confidence detections
├── ⚡ optimized_models/     # ONNX & TensorRT models
├── 📈 benchmarks/           # JSON results & charts
└── 📊 charts/               # Performance visualizations
```

---

## 🎯 Requirements Compliance

| Requirement | Target | Status |
|:-----------:|:------:|:------:|
| Real-Time FPS | ≥ 25 FPS | ✅ **PASS** |
| Detection Accuracy | ≥ 80% mAP | ✅ **PASS** |
| Model Size | ≤ 150 MB | ✅ **PASS** |
| Inference Latency | < 50 ms | ✅ **PASS** |

---

## 🔮 Future Improvements

- [ ] 🎯 INT8 Quantization for further optimization
- [ ] 📱 Edge device deployment (Jetson Nano/Xavier)
- [ ] 🔄 Real-time streaming integration
- [ ] 🌐 Web-based dashboard
- [ ] 🤖 Multi-object tracking (DeepSORT)

---

## 📚 References

- [Ultralytics YOLOv8](https://docs.ultralytics.com/)
- [Microsoft AirSim](https://microsoft.github.io/AirSim/)
- [CARLA Simulator](https://carla.org/)
- [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt)

---

<div align="center">

### 🌟 Star this repo if you find it useful!

Made with ❤️ for Edge AI Research

[![GitHub Stars](https://img.shields.io/github/stars/username/edge-ai-drone-detection?style=social)](https://github.com/username/edge-ai-drone-detection)

</div>
