# Hexagon DSP 开发教程

从零开始学习 Qualcomm Hexagon DSP 编程，覆盖 HVX（向量处理）和 HMX（矩阵加速）。

## 目录

- [第一章：安装模拟器，跑通 HVX + HMX](ch01-simulator-setup/) — x86 Linux 模拟器
- [第二章：在真机上跑 HVX + HMX](ch02-real-device/) — 骁龙 8 Gen 3 CDSP
- [第三章：QNN 自定义算子 (HVX+HMX)](ch03-qnn-custom-op/) — QHPI Custom Op

## 环境要求

- x86_64 Linux + Hexagon SDK 6.4.0.2
- 第二、三章额外需要：骁龙 8 Gen 3（或更新）手机 + adb
- 第三章额外需要：QNN SDK (QAIRT) 2.44+

## 快速开始

```bash
# 安装工具 (首次)
bash ch01-simulator-setup/install_tools.sh

# 第一章: 模拟器
bash ch01-simulator-setup/run.sh

# 第二章: 真机
bash ch02-real-device/build.sh
bash ch02-real-device/run_device.sh

# 第三章: QNN 自定义算子
bash ch03-qnn-custom-op/build.sh
bash ch03-qnn-custom-op/run_device.sh
```
