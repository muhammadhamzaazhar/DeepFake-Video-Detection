# Deepfake Video Detection System using Multi-Modality Features

This repository contains the code and resources for the final year thesis project titled **"Development of DNN based Deepfake Video Detection system using multi-modality features,"** submitted in partial fulfillment of the requirements for the **Bachelor of Sciences in Computer and Information Sciences at the Pakistan Institute of Engineering and Applied Sciences (PIEAS).**

The project, aims to detect deepfake videos by leveraging both **optical artifacts** and **Transdermal Optical Imaging (TOI)** features through a deep learning approach.

We utilized and reproduced concepts from the following research papers:

- [**Deepfake Video Detection Using Convolutional Vision Transformer**](https://arxiv.org/abs/2102.11126)
- [**TALL: Thumbnail Layout for Deepfake Video Detection**](https://openaccess.thecvf.com/content/ICCV2023/papers/Xu_TALL_Thumbnail_Layout_for_Deepfake_Video_Detection_ICCV_2023_paper.pdf)

[![Thesis Report](https://img.shields.io/badge/Thesis_Report-blue?style=for-the-badge)](https://drive.google.com/drive/folders/1jR4267im695K5J5Tm--nccRdyyhRl8z1?usp=drive_link)
[![Presentations](https://img.shields.io/badge/Presentations-008000?style=for-the-badge)](https://drive.google.com/drive/folders/1kg8cgnT1pfNLuyHXs3P_MQ6qC2iHof18?usp=drive_link)

![Web-App](https://github.com/user-attachments/assets/6e03203c-9422-4cce-87fe-edf231222a72)

---

## Table of Contents

- [Project Description](#project-description)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Technical Details](#technical-details)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

---

## Project Description

Deepfake technology poses a significant threat to media credibility and cybersecurity by creating highly realistic falsified videos that can:

- Spread misinformation
- Damage reputations
- Facilitate crimes such as fraud and propaganda

This project addresses the challenge of detecting deepfakes by combining:

- **Optical Artifacts** (e.g., facial landmark distortions, unnatural movements)
- **TOI Features** (e.g., subtle blood flow patterns beneath the skin)

A **Deep Neural Network (DNN)** approach is used to analyze these **multi-modality features** to provide a robust and accurate detection system.

A web application complements the backend, offering real-time video authenticity verification.

---

## Features

- **Multi-Modality Feature Extraction**: Combines optical and TOI features.
- **Thumbnail Layout (TALL)**: Efficient video representation using compact thumbnails.
- **Deep Learning Models**:
  - CViT (Convolutional Vision Transformer)
  - TALL-TimeSformer (Transformer with thumbnail input)
- **Web Application**: Real-time video analysis via browser-based interface.

---

## Technologies Used

- **Programming Language**: Python 3.10
- **Frameworks**:
  - PyTorch (Deep Learning)
  - Next.js (Frontend)
- **Database & Hosting**:
  - Supabase (Storage/DB)
  - Modal (Model Hosting)
  - Vercel (Frontend Deployment)
- **Other Packages**: MTCNN, NumPy, Transformers, OpenCV, WandB, Facenet-PyTorch, Deepspeed

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/muhammadhamzaazhar/DeepFake-Video-Detection.git
cd DeepFake-Video-Detection
```

### Install Dependencies

- Make sure **Python 3.10** is installed on your system.
- Then, install the required Python packages by running:

```bash
pip install -r requirements.txt
```

To make a prediction on a video:

1. Download the pretrained model and `config.js` from [Hugging Face](https://huggingface.co/mhamza-007/timesformer-deepfake-detection/tree/main).
2. Place the downloaded files inside the `weights/` directory.
3. Run the prediction script using:

```bash
python predict.py "path/to/your/video.mp4"
```

---

## Technical Details

For complete details on:

- Dataset sources and structure
- Preprocessing pipeline (face detection, skin segmentation, TOI heatmaps)
- Model architecture and training methodology

please refer to the [Thesis Report](https://drive.google.com/file/d/1R4JDMLhXBRmX9okUZ6fTWcQQTcB2t2TQ/view?usp=sharing).

---

<a name="acknowledgements"></a>

## Acknowledgements

- **Supervisor**: Dr. Asifullah Khan, DCIS PIEAS
- **Co-Supervisor**: Dr. Abdul Majid, DCIS PIEAS
- **Research Labs**:
  - PRLab
  - PIEAS AI Center (PAIC)
  - Center for Mathematical Sciences (CMS)

---

<a name="contact"></a>

## Contact

For inquiries or feedback, please reach out to:

### Muhammad Hamza Azhar

- Email: [hamzaazharmuhammad@gmail.com](mailto:hamzaazharmuhammad@gmail.com)
- LinkedIn: [https://www.linkedin.com/in/muhammad-hamza-azhar](https://www.linkedin.com/in/muhammad-hamza-azhar-996289314/)

### Haider Ali Aurangzaib

- Email: [haiderali10986@gmail.com](mailto:haiderali10986@gmail.com)
- LinkedIn: [https://www.linkedin.com/in/haider-ali-aurangzaib](https://www.linkedin.com/in/haider-ali-aurangzaib/)
