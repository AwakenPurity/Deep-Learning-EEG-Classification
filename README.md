# Deep-Learning-EEG
基于深度学习的脑电图（EEG）睡眠状态检测

数据集：Sleep-EDF Database Expanded
- 链接：https://www.physionet.org/content/sleep-edfx/1.0.0/
> "Sleep-EDF Database Expanded"（扩展的Sleep-EDF数据库）是一个用于睡眠研究的数据集。原始的Sleep-EDF数据库是由PhysioNet提供的，包含了多个参与者的多通道生理信号数据，用于研究睡眠相关的信息。扩展的Sleep-EDF数据库是在原始数据库(https://www.physionet.org/content/sleep-edf/1.0.0/) 基础上进行扩展和改进的版本。它包含更多的参与者数据和更多的生理信号通道。这些信号通道包括脑电图（EEG）、眼动图（EOG）、肌电图（EMG）和心电图（ECG），以及其他相关的生理信号。这个数据集的目的是为睡眠研究提供更多的多模态数据，以帮助研究人员深入理解睡眠过程和相关的生理现象。研究者可以使用这个数据集来开发和评估算法，进行睡眠分析、睡眠阶段分类、睡眠障碍诊断等相关研究。

本实验中，主要将睡眠状态分为五种类型：
- Wake（清醒）：代表觉醒状态，即醒着且清醒的状态。
- NREM（Non-Rapid Eye Movement）睡眠：代表非快速眼动睡眠，包括三个阶段（N1、N2、N3）。
- REM（Rapid Eye Movement）睡眠：代表快速眼动睡眠，一种睡眠状态，通常与活跃的梦境发生关联。
