---
layout: post
title: Linux机器上安装和配置nvidia-docker训练环境操作指南
date: 2024-04-25 15:49:00 +0800
description: 介绍如何基于nvidia-docker，在Linux机器上安装和配置用于训练深度学习模型的环境
tags: [Machine Learning]
relatedposts: false
---

Docker可以提供相互独立的训练环境，一方面可以实现不同用户训练环境的隔离，另一方面docker镜像可以被复用，以避免重复的环境搭建或者不同机器上环境不一致的问题。因此，在深度学习方向的开发机上使用docker是一个较好的实践。

本文介绍在完成NVIDIA-driver的安装后，安装nvidia-docker的过程。

### 1. 安装docker
可以通过下方命令安装docker
```bash
curl https://get.docker.com | sh \
  && sudo systemctl --now enable docker
```

安装完成后，可以通过以下命令，确认docker已经安装成功。
```bash
sudo docker version
```

### 2. 配置nvidia源
可以通过下方命令配置nvidia源
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
      && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

### 3. 安装nvidia-docker
可以通过下方命令安装nvidia-docker

```bash
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
```

安装完成后，配置docker的运行环境，并重启docker
```bash
sudo nvidia-ctk runtime configure --runtime=docker
```

```bash
sudo systemctl restart docker
```

### 4. 测试安装成功
配置和重启完成后，可以通过下方命令新建容器并进行测试：
```bash
sudo docker run --rm --runtime=nvidia --gpus all nvidia/cuda:11.6.2-base-ubuntu20.04 nvidia-smi
```
如果安装顺利完成，终端上将打印出机器上的显卡信息。