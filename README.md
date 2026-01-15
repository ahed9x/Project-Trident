## Project: Cairo Smart-Trident 🔱

I am on a mission to create the most reliable open-source printer based upon voron printers' design, with plug and play and auto error detection for nearly everything.

## Project Overview

High-end 3D printer that uses many smart, low-cost components and mainly a toolhead-mounted endoscope camera to possibly match or even beat the Bambu Labs AI systems using AI and smart vision aproaches democratizing industrial smart printing using mostly opensource widely available electronics; furthermore, it may use other sensors to achive this goal.

## 🧑‍🔬 Key Engineering Innovations

### 📷 1. The "Side-Car" Metrology Mount (Custom CAD)

Unlike standard StealthBurner modifications that disrupt airflow, I designed a parametric external mounting bracket using the existing models and slots available for an ADXL accelometer mount.

* **Design Intent:** Integrates a 5.5mm endoscope for computer vision without compromising thermal aerodynamics or blocking the inductive probe.
* **Feature:** Includes an M3 grub-screw locking mechanism for vibration resistance during high-speed CoreXY motion.
* **File:** [`/CAD/Trident_Nozzle_Cam_Mount_v1.step`](https://github.com/ahed9x/Project-Trident/blob/main/CAD/Trident_Nozzle_Cam_Mount_v1.step)

### 🤖 2. Frugal AI & "Edge" Optimization

Most comercially avaliabe 3d printers use local NPU for heavy AI workload thats why I am creating these paths:

#### Path 1:
* **Hardware:** Standard Raspberry Pi 3 (Cost-effective).
* **Software:** Optimized TFLite models running snapshot-based inference (every 10-15s) rather than real-time video stream processing.
* **Result:** Reliable spaghetti detection and flow calibration without requiring expensive hardware.

#### Path 2:
* **Hardware:** Orange Pi 5 Pro 4gb.
* **Software:** slightly higher, more dedicated AI models as the Orange Pi has faster cores and 6 built-in AI tops (3x the Bambu Lab X1c)
* **Results:** Extremely accurate AI models that are needed for auto calibration and even real-time compensation at at least 15 fps

#### Path 3:
* **Hardware:** Any
* **Software:** it will lack real time anything except if used apis but calibrationion process will be calculated on the users own pc 
* **Results:** Extremely cost-efficient but relies on the users pc to be on while calibrating

### ⛓️ 3. Hybrid Supply Chain (Cost Engineering)

To fit the budget, the machine might be down-sized to the **250mm spec**:

* **Localization:** everything is sourced locally from my home country -Egypt- except for some bearings and the Trianglelab CHC Pro hotend.
* **Strategic Imports:** Only scarce or unavailable components are imported via AliExpress.

---

## ⚙️ Technical Specifications

* **Kinematics:** CoreXY (Belt Driven)
* **Build Volume:** 300mm x 300mm x 250mm
* **Firmware:** Klipper + Moonraker
* **Toolhead:** Voron StealthBurner + Clockwork 2
* **Electronics:** BTT Octopus / CAN-bus via EBB SB2209
* **Vision:** 5.5mm Endoscope (USB)

## 💸 Bill of Materials (BOM) & Budget

I have optimized the BOM, but it is still not 100% ready yet.

* [📄 Click here to view the full BOM.csv](https://docs.google.com/spreadsheets/d/1iS4H5-gJYytW3T8kcJbArUC25VIOrDTH6kZ6Q5hKLfA/edit?usp=sharing)

## 📅 Execution Timeline

* **Week 1:** ordering and buying + cutting.
* **Week 2:** Frame assembly and squaring.
* **Week 3-5:** wiring everything and assembling other parts.
* **Week 6:** "Side-Car" mount printing and AI Model testing/tuning, and researching more options and sensors.

---

## 📸 Visuals

<img width="1524" height="838" alt="image" src="https://github.com/user-attachments/assets/312d6bb5-40e2-44a4-a674-da3441a020c2" />
<img width="841" height="692" alt="image" src="https://github.com/user-attachments/assets/745d6112-8bf4-407c-838d-f574df7e7d74" />



---
Note: I wrote most of this and even added emojis manually
