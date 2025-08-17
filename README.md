# 🧭 Explore Package – Modified from `explorer_lite`

Welcome! This is a modified version of the `explorer_lite` package for ROS 2. I’ve customized it to improve the autonomous exploration experience, especially when working with **RTAB-Map** and **Nav2**.

This package includes some new features like:
- An **ArUco marker detector**
- A **boundary generator**
- Ready-to-use **launch and config files** for both mapping and navigation

It's been tuned to work with the **Panther robot from Husarion**, but you can easily adapt it for your own robot by adjusting a few parameters.

---

## 🚧 Work in Progress

Just a heads-up: the **ArUco marker detector** and **boundary generator** are still under development. The marker positions detected might not be accurate yet — they tend to **deviate from the actual position**, so treat them as experimental for now.

---

## ✅ What's Included

- 🧭 Base exploration logic (from `explorer_lite`)
- 🆕 ArUco marker detector (WIP – needs tuning)
- 🆕 Boundary generator (WIP)
- ⚙️ Parameter files for **Nav2** and **RTAB-Map**
- 🚀 Launch files to run everything quickly
- 🦾 Speed profile designed for **Panther Husarion**, but easy to tweak

---

## 📦 How to Install

### Step 1: Clone this into your ROS 2 workspace

```bash
cd ~/your_ros2_ws/src
git clone https://github.com/kruthikhanurajkumar/Explore_by_roboconiitr
```
### Step2:Install dependencies and build

```bash
cd ~/your_ros2_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

## 🚀 How to Use
### Launch RTAB-Map + Nav2 with exploration:

```bash
ros2 launch explore slam_and_nav_setup.launch.py
```
### Launch aruco_marker_detector:

```bash
ros2 launch explore aruco.launch.py
```


