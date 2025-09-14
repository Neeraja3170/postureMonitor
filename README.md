# PostureGuard AI

A **real-time posture correction system** built with **OpenCV** and **MediaPipe**.
The project detects your posture using a webcam and provides **instant feedback** to help maintain a healthy sitting position.

---

## 🖥️ System Requirements

* **Operating System**: Windows 10/11, Linux, or macOS
* **Python Version**: 3.8 – 3.11 (recommended: Python 3.10)
* **Hardware**:

  * Webcam (built-in or external)
  * Minimum 4GB RAM
  * Processor: Intel i3 / AMD Ryzen 3 or higher

---

## 📦 Requirements

Install all dependencies using:

pip install -r requirements.txt


## ▶️ Run the Project

After installing the requirements, start the program with:
python posture_guard.py


## 💡 Main Concept

The project uses **MediaPipe Pose Estimation** + **OpenCV** to:

* Detect human posture in **real time** using a webcam.
* Track key body landmarks like **shoulders, neck, and spine alignment**.
* Give **audio/visual feedback** if poor posture is detected (via `playsound`).

This helps reduce back/neck strain from long working hours and promotes better ergonomic health.

---

## 📸 Demo Instructions

1. Run the program (`python main.py`).
2. Your **webcam feed** will open.
3. Sit in front of the camera.

   * ✅ Correct posture → No alert.
   * ❌ Bad posture → An **audio alert** will play.
4. Press **Q** to quit the program.

---

## 🖼️ Example Usage

### Good Posture

![Image](https://github.com/user-attachments/assets/c3a4b0a8-2651-4a52-b2dd-9b1341b7c339)

### Bad Posture Alert

![Bad Posture Example](https://via.placeholder.com/500x300?text=Bad+Posture+Alert+Sound+Played)
