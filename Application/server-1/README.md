# Backend Server for Deepfake Video Detection

The backend server for this project is **hosted on Hugging Face Spaces** and is fully functional online.

## 🚀 Live Server

You don't need to run the backend locally — it’s already deployed and available via a public API.

👉 **[Access the Backend Server on Hugging Face](https://huggingface.co/spaces/mhamza-007/deepfake-video-detection/tree/main)**

---

## 📦 What's Inside This Folder?

This folder acts as a reference. It does **not contain server code**.

The actual backend:

- Uses Python (FastAPI)
- Is built to load the CViT deepfake detection model
- Performs inference on uploaded videos
- Returns prediction results (REAL / FAKE) with confidence

---

## 📝 If You Still Want to Run It Locally

Clone the full backend from the Hugging Face-hosted repo:

```bash
git clone https://huggingface.co/spaces/mhamza-007/deepfake-video-detection
cd deepfake-video-detection

```
