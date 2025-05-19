USE CASE : Detecting and describing every object in a frame
1) Generate multiple binary masks from an image using SAM and filter them by
confidence threshold.
We start with generating segmentation masks using the Segment Anything Model (SAM).
SAM returns a list of masks along with a confidence score for each mask. We retain only
those masks which exceed a certain threshold (e.g.0.7) to eliminate low-quality predictions:
masks = sam.predict(image)
M = [m[&#39;segmentation&#39;] for m in masks if m[&#39;score&#39;] &gt; 0.7]
Each mask is a binary segmentation map representing the region of the object in the image.
2) Now we often have a single object being split into multiple masks. To counter it we have
to merge those pieces into a unified mask per object.
a) Object Detection Guided Merging using OWL-ViT (Preferred)
Instead of YOLO, which is limited to COCO’s 80 pre defined classes, we use YOLOv8 World, an
open vocabulary object detector capable of detecting objects from arbitrary categories
using natural language prompts. This enables flexibility and zero-shot detection, which is
important when working with diverse and unseen object types in the image.
- Run YOLO World on the image to obtain bounding boxes for objects using category prompts.
- Loop through all binary masks.
- For each detected bounding box from YOLO World, collect masks that significantly overlap
with it.
- Merge those masks into a single unified mask using `cv2.bitwise_or`.
merged_mask = np.zeros_like(image[:,:,0])
for m in overlapping_masks:
merged_mask = cv2.bitwise_or(merged_mask, m)
This approach ensures masks corresponding to the same object are grouped based on their
spatial location relative to the detected bounding boxes.

3) Object Cropping
Once we have a merged mask corresponding to a single object, we proceed to crop the
relevant region from the original image.
a) Find the bounding box:
Use OpenCV to find the smallest rectangle enclosing all non-zero pixels in the binary mask:
x, y, w, h = cv2.boundingRect(merged_mask.astype(np.uint8))
b) Crop the region from the original image:
Extract the rectangular region of interest (ROI) containing the object:
cropped = image[y:y+h, x:x+w]
c) Optional Padding/Resize:
To maintain uniformity in input dimensions (especially if required by the captioning
model), we may optionally pad or resize the cropped object.
4) Feed Cropped Objects to DAM
We use a Description Auto-Model (DAM) to generate captions for each cropped object. The
DAM could be based on:
- BLIP-2
- LLaVA-1.5

These models are capable of understanding the visual content of the cropped object and
generating meaningful natural language descriptions.

description = dam.generate(cropped)

5) Post Processing
Once we have raw captions generated for the objects, we post-process them in two stages:
1) Counter duplication
Often, visually similar or identical objects will result in repeated or near-identical captions.
To remove these:
from sklearn.metrics.pairwise import cosine_similarity
embs = clip.encode_text(descriptions)
similarity_matrix = cosine_similarity(embs)
- Remove captions with cosine similarity above a threshold (e.g., 0.9).
2) LLM Refinement
To improve fluency, grammar, and conciseness of the captions, we use GPT-4:
prompt = f&quot;Improve this description: &#39;{description}&#39;. Be concise and precise.&quot;
refined_desc = openai.ChatCompletion.create(
model=&quot;gpt-4&quot;,
messages=[{&quot;role&quot;: &quot;user&quot;, &quot;content&quot;: prompt}]
)
This ensures that the final output descriptions are professional, accurate, and suitable for
downstream use such as digital asset management or accessibility tools.
Conclusion:
This pipeline enables robust object-level captioning by combining SAM (for segmentation),
YOLO World (for detection), and DAM (for description). Unlike YOLO, YOLO World's zero-shot
capability gives us the flexibility to detect and describe objects beyond COCO’s fixed class
set, making it more suitable for general-purpose and open-domain applications.

