# 🎯 Object-Level Captioning Pipeline

This project demonstrates an end-to-end pipeline for detecting and describing **every object in an image frame**. It combines state-of-the-art models including **SAM**, **YOLO World (v8)**, **BLIP-2/LLaVA**, and **GPT-4** to produce **object-level image captions** with high precision and generalizability.

---

## 📌 Use Case

This pipeline enables:
- **Fine-grained object detection and segmentation**
- **Flexible object recognition** beyond fixed COCO classes
- **Zero-shot visual captioning**
- **Useful for accessibility, digital asset management, search indexing, and more**

---

## 🧠 Pipeline Overview

### 1. 🔍 Segmentation with SAM
We use [Meta's Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) to generate binary segmentation masks for all visible objects.

```python
masks = sam.predict(image)
filtered_masks = [m['segmentation'] for m in masks if m['score'] > 0.7]
```

Only high-confidence masks (score > 0.7) are retained.

---

### 2. 🧩 Mask Merging with YOLO World (Object Detection)
To handle fragmented masks of a single object:
- We run **YOLOv8 World**, an open-vocabulary object detector.
- For each detected bounding box, we group and merge all overlapping masks.
- Merging is performed using OpenCV:

```python
merged_mask = np.zeros_like(image[:, :, 0])
for m in overlapping_masks:
    merged_mask = cv2.bitwise_or(merged_mask, m)
```

YOLO World enables zero-shot detection using text prompts, unlike traditional YOLO models.

---

### 3. ✂️ Object Cropping
Once a unified mask is created:
- Compute bounding box with OpenCV:
```python
x, y, w, h = cv2.boundingRect(merged_mask.astype(np.uint8))
```
- Crop object from the original image:
```python
cropped = image[y:y+h, x:x+w]
```
- Optionally resize/pad for captioning model input compatibility.

---

### 4. 🧾 Captioning with Description Auto-Model (DAM)
Each cropped object is passed to a visual-language model such as:
- **[BLIP-2](https://huggingface.co/Salesforce/blip2)**
- **[LLaVA-1.5](https://github.com/haotian-liu/LLaVA)**

Example:
```python
description = dam.generate(cropped)
```

---

### 5. 🧹 Post-Processing

#### 1) **Remove Duplicates**  
Using cosine similarity on CLIP embeddings to eliminate near-identical captions:
```python
embs = clip.encode_text(descriptions)
similarity_matrix = cosine_similarity(embs)
```

#### 2) **Refine Captions using GPT-4**
Improve language fluency and clarity:
```python
prompt = f"Improve this description: '{description}'. Be concise and precise."
refined_desc = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}]
)
```

---

## ✅ Advantages

- **Flexible vocabulary** with YOLO World
- **Fine-grained, high-quality captions** via DAM + GPT-4
- **Applicable in zero-shot settings**
- **Modular design for easy customization**

---

## 📁 Folder Structure

```
.
├── segment.py       # SAM-based mask generation
├── detect.py        # YOLO World detection
├── merge.py         # Mask merging logic
├── crop.py          # Object cropping module
├── caption.py       # BLIP-2 / LLaVA caption generation
├── refine.py        # GPT-4 post-processing
├── utils/           # Helper functions
└── README.md        # This file
```

---

## 🔧 Requirements

- Python 3.8+
- OpenCV
- NumPy
- PyTorch
- HuggingFace Transformers
- `ultralytics` for YOLOv8 World
- OpenAI GPT-4 API access
- CLIP for duplicate detection

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 🚀 Run the Pipeline

```bash
python main.py --image input.jpg
```

---

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🙌 Acknowledgments

- Meta AI for Segment Anything
- Ultralytics for YOLO World
- Salesforce for BLIP-2
- LLaVA Team
- OpenAI for GPT-4
