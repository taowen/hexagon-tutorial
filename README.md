# Hexagon DSP 开发教程

从零开始学习 Qualcomm Hexagon DSP/NPU 编程。两条独立的学习路线，共享前两章环境搭建：

| | [HMX 教程](hmx-tutorial/) | [QNN 教程](qnn-tutorial/) |
|---|---|---|
| **定位** | 直接编程 HVX/HMX 硬件 | 通过 QNN 框架开发 |
| **内容** | 通信、内存、矩阵乘法、KV Cache、llama.cpp、DSP 训练 | 自定义算子、x86 模拟、VTCM 管理、Genie SDK |
| **适合** | 想深入理解 NPU 底层机制的开发者 | 想快速在 NPU 上部署模型的应用开发者 |
| **章节** | 共同基础 2 章 + 正文 6 章 | 共同基础 2 章 + 正文 4 章 |

## 共同基础

- **[第一章：安装模拟器，跑通 HVX + HMX](ch01-simulator-setup/)** — 在 x86 Linux 上配置 Hexagon SDK 6.4，用模拟器运行第一个 HVX + HMX 程序
- **[第二章：在真机上跑 HVX + HMX](ch02-real-device/)** — 通过 FastRPC 部署到骁龙 8 Gen 3，申请 VTCM/HVX/HMX 硬件资源
