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

https://private-user-images.githubusercontent.com/143608295/489236421-491cc6e6-7172-4b19-b4a9-434dc5a2e349.jpg?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NTc4MjY0NDYsIm5iZiI6MTc1NzgyNjE0NiwicGF0aCI6Ii8xNDM2MDgyOTUvNDg5MjM2NDIxLTQ5MWNjNmU2LTcxNzItNGIxOS1iNGE5LTQzNGRjNWEyZTM0OS5qcGc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjUwOTE0JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI1MDkxNFQwNTAyMjZaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1hMjM0MTgyYmU4MzM5M2EyMzc3OGE2MDE1ZjkxODgxNzkyNTMxNjIzOTAzMDhhMWJjNTU4NTFjYTg5OGFjN2Y4JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.tk3Dc4gEBZrHNp8I-vJeTwDdTzom_gmi-iYQ2VNTg3w

### Bad Posture Alert

![Bad Posture Example](https://via.placeholder.com/500x300?text=Bad+Posture+Alert+Sound+Played)
