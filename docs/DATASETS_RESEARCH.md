# Datasets Research for ServeSense

Datasets that are useful for volleyball serve analysis (pose, ball detection, benchmarks) and how easy they are to implement.

---

## 1. Ball detection (improve YOLO for volleyball)

| Dataset | Description | Format | Ease | Link / Notes |
|--------|-------------|--------|------|--------------|
| **Volleyball Tracker (Roboflow)** | 838 images, ball detection, pre-trained YOLOv11 | YOLO (v5/v8/v11 export) | **Easy** | [Roboflow Universe](https://universe.roboflow.com/volleyballnft/volleyball-tracker-vvvdx). Free account → Export as YOLOv8 → use Python SDK or CLI to download. |
| **Volleyball Activity Dataset (Roboflow)** | ~25k images, multi-class (activity + ball) | YOLO, COCO | **Easy** | [Roboflow](https://universe.roboflow.com/activity-graz-uni/volleyball-activity-dataset). Export in YOLO format; filter classes for “ball” if needed. |
| **Volleyball Ball Object Detection (GTS.ai)** | Volleyball locations, 640×640, from 22s match clip | YOLOv8x-oriented | **Medium** | [GTS.ai](https://gts.ai/dataset-download/volleyball-ball-object-detection-dataset/). Download and convert to same format as your trainer if needed. |
| **ai volley detection (Roboflow)** | 782 images, CC BY 4.0 | YOLO export | **Easy** | [Roboflow](https://universe.roboflow.com/volleyball-ai-detection/ai-volley-detection). Same workflow as above. |

**Easiest path:** Use Roboflow’s Python SDK: install `roboflow`, set `ROBOFLOW_API_KEY`, run the provided script to download a volleyball ball dataset in YOLO format, then fine-tune or evaluate your current ball detector (see `scripts/download_roboflow_volleyball.py`).

---

## 2. Pose / activity (validate or extend pose on volleyball)

| Dataset | Description | Format | Ease | Link / Notes |
|--------|-------------|--------|------|--------------|
| **SportCenter (EPFL)** | Volleyball sequences with pose, homography, player positions | Custom (likely JSON + images) | **Medium** | [EPFL SportCenter](https://www.epfl.ch/labs/cvlab/data/sportcenter-dataset/). Good for validating pose on volleyball; may need a small loader. |
| **Volleyball Activity Dataset 2014** | 6 HD videos, 7 activity classes (Serve, Reception, Set, Attack, etc.) | Annotations per frame | **Medium** | [MLDTA](https://mldta.com/dataset/volleyball-activity-dataset-2014/). Useful for serve vs non-serve classification, not raw keypoints. |
| **VREN** | Large-scale rally footage, player actions/positions | Expression notation, GitHub | **Medium** | [VREN (arXiv)](https://arxiv.org/html/2209.13846v2). Tactical/rally focus; check repo for exact annotation format. |

**Easiest path:** Keep using COCO-pretrained YOLOv8-pose; optionally add SportCenter or Volleyball Activity as a validation set to run your current pipeline on and compare.

---

## 3. Event timing & serve segments (benchmark / evaluation)

| Dataset | Description | Format | Ease | Link / Notes |
|--------|-------------|--------|------|--------------|
| **VNL-STES (VNL Dataset)** | 8 full matches, 1,028 rallies, 6,137 events (incl. 1,071 Serves) with **frame + (x,y)** | Rally videos + annotations (temporal + spatial) | **Medium** | [Project page](https://hoangqnguyen.github.io/stes/) · [Data (bit.ly)](https://bit.ly/vnlvolley1). Request access; use Serve events for “ground truth” serve timing and position to evaluate your detector. |
| **VNL-STES splits** | Train 811 / Val 102 / Test 115 rallies | Same as above | **Medium** | Same download. Use test set to report metrics. |

**Easiest path:** Once you have the VNL data, write a small loader for their annotation format and extract Serve events (frame index + optional x,y) to compare against your pipeline’s contact/toss frames.

---

## 4. Reference metrics (pro benchmarks from literature)

| Source | What it gives | Ease | Use in ServeSense |
|--------|----------------|------|-------------------|
| **Kinematic studies (e.g. spatio-temporal serve analysis)** | Contact height (MHTC), elbow timing, phase durations | **Easiest** | No video needed. Manually enter ranges or point estimates into `pro_benchmarks.json` or `literature_benchmarks.json` (see below). |
| **High-level women’s volleyball (Degruyter)** | Serve type, effectiveness, some performance factors | **Easy** | Qualitative; supports which serve types to include (e.g. jump float, standing float). |
| **Sitting volleyball (serve efficacy)** | Win/loss by serve outcome | **Easy** | Optional “reference” context; not needed for metric comparison. |

**Easiest path:** Use `backend/app/data/literature_benchmarks.json` (provided in this repo). It includes a few reference entries with plausible values from published studies so you always have at least one benchmark per serve type/camera even before adding your own pro clips.

---

## 5. Pro benchmark data sources (for tuning comparisons)

Papers and datasets with actual kinematic values. **Note:** Definitions differ across studies. Our metrics are:
- **toss_height_m**: Ball apex height (m)
- **contact_height_m**: Vertical distance from top of head to hand at contact (m)
- **elbow_height_m**: Vertical distance from top of head to elbow at contact (m)
- **toss_contact_time_s**: Time from ball release to contact (s)

### Published studies with extractable metrics

| Paper | Data | Link | Notes |
|-------|------|------|-------|
| **Spatio-temporal Analysis of Volleyball Serve (IJEDR 2019)** | 15 Indian international male players. Jump float: MHTC 312±47 cm (toss), SSHN 280±15 cm (contact height). Jump topspin: MHTC 655±126 cm, SSHN 321±43 cm. | [PDF](https://rjwave.org/ijedr/papers/IJEDR1903140.pdf) | MHTC = Max Height Gain of Toss; SSHN = Serve Spike Height (absolute). Our head-to-hand is body-relative. |
| **Evaluation of Two Methods of the Jump Float Serve (JAB 2012)** | Jump float serve techniques: contact height, ball speed, projection angle. Jump-focused vs contact-focused. | [Human Kinetics](https://journals.humankinetics.com/view/journals/jab/28/5/article-p579.xml) | Contact height and technique comparison. |
| **Upper Limb Biomechanics During Volleyball Serve (PMC 2012)** | 3D kinematics, humerus-trunk angles, hand speed. Elite players. | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC3445065/) | Angular/velocity data; need conversion for our metrics. |
| **An Approach to Quantify Float Effect (Frontiers 2020)** | Float serve trajectory, indoor vs beach. | [Frontiers](https://www.frontiersin.org/articles/10.3389/fspor.2020.559277/pdf) | Ball trajectory; less on body metrics. |
| **Analysis of kinematic variables predicting jump serve efficacy (Bari et al. Medicine 2023)** | 30 male collegiate players. U-23: spike 327.4±3.8cm, elbow 133°, shoulder 71° at contact. U-21: spike 268.8cm, U-19: 287.6cm. | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10402994/) | Full Table 2 with spike height, elbow/shoulder/wrist angles. Approach velocity, SEACP, CG predict ball velocity. |
| **Three-dimensional kinematic analysis of float serve (Cal State thesis)** | Division I male players. Elbow 112–117°, shoulder 114–121° at contact. Short vs long serve. | [ScholarWorks](https://scholarworks.calstate.edu/concern/theses/8c97kv69z) | Float-specific angles; lower than jump serve. |
| **Three-dimensional kinematic analysis of volleyball jump serve (Konstanz)** | Coleman 1997 thesis data. Elbow/humerus extension. | [OJS](https://ojs.ub.uni-konstanz.de/cpa/article/view/3718) | Classic reference; may need request. |

### How to add new benchmarks

1. Extract values from a paper (convert units: cm → m).
2. Map to our schema: if paper reports *absolute* contact height (e.g. SSHN), estimate head-to-hand as ~0.15–0.45 m (typical arm extension above head). Or use create-benchmark with pro screenshots.
3. Add to `backend/app/data/literature_benchmarks.json` (see existing entries for schema).
4. Restart backend to load new benchmarks.

---

## Implementation summary

| Goal | Easiest option | Effort |
|------|----------------|--------|
| Better ball detection on clear videos | Roboflow “Volleyball Tracker” or “Volleyball Activity” → download in YOLO → fine-tune or evaluate | 1 script + API key |
| More pro benchmarks without new video | Add `literature_benchmarks.json` and load it in the app | Already added; wire into API if desired |
| Ground-truth serve timing for evaluation | VNL-STES (request data) → parse Serve events → compare to your pipeline | 1 loader script + data request |
| Pose validation on volleyball | SportCenter or Volleyball Activity 2014 → run current YOLOv8-pose on their videos | 1 loader + evaluation script |

---

## Files added in this project

- **`docs/DATASETS_RESEARCH.md`** (this file) – summary and links.
- **`backend/app/data/literature_benchmarks.json`** – reference benchmarks from literature (same schema as `pro_benchmarks.json`); loaded automatically with pro benchmarks.
- **`scripts/README.md`** – step-by-step for dataset download and ball-model evaluation.
- **`scripts/download_roboflow_volleyball.py`** – download a Roboflow volleyball dataset in YOLO format (requires `roboflow` and `ROBOFLOW_API_KEY`).
- **`scripts/evaluate_ball_model.py`** – run validation (and optional training) on the downloaded dataset.
- **`scripts/requirements-datasets.txt`** – optional deps: `pip install -r scripts/requirements-datasets.txt`.
