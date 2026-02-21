🌾 AgriYouth: AI-Powered Plant Disease Detection for African Youth
https://img.shields.io/badge/python-3.12-blue.svg
https://img.shields.io/badge/PyTorch-2.10.0-ee4c2c.svg
https://img.shields.io/badge/TensorFlow-2.19.0-ff6f00.svg
https://img.shields.io/badge/License-MIT-yellow.svg
https://img.shields.io/badge/Open%2520in-Colab-F9AB00.svg

📋 Table of Contents
Mission & Vision

Project Overview

Key Findings

Repository Structure

Setup Instructions

Dataset

Model Architectures

Experiments & Results

Mobile Deployment Considerations

Reproducibility

Report & Demo

License

Author

Acknowledgments

🎯 Mission & Vision
Mission: Empower 1 million African youth by 2030 with AI-driven agricultural tools that transform farming from subsistence to profitable, tech-enabled entrepreneurship.

Vision: A future where every young farmer in Africa has access to AI-powered diagnostic tools on their mobile phones, enabling early disease detection, reduced crop losses, and increased agricultural productivity.

📖 Project Overview
AgriYouth is a machine learning project that develops and evaluates deep learning models for automated plant disease diagnosis. The project systematically compares five distinct approaches:

#	Model	Description
1	ResNet-50	Baseline model for high-accuracy benchmark
2	MobileNetV3	Mobile-optimized architecture for entry-level phones
3	EfficientNet-B0	Balanced model for mid-range devices
4	AgriYouthNet	Custom ultra-lightweight architecture
5	AgriYouthNet-Context	Novel context-aware model with farmer metadata
The models are trained on the PlantVillage dataset and evaluated on key metrics for mobile deployment: accuracy, model size, training time, and inference efficiency.

🔬 Key Findings
🏆 Rank	Model	Size (MB)	Accuracy (%)	Key Insight
1st	AgriYouthNet-Context	0.58	99.06%	🏆 BEST OVERALL - Context awareness + ultra-lightweight
2nd	EfficientNet-B0	15.34	98.48%	Highest base accuracy, good for mid-range phones
3rd	ResNet-50	89.75	97.75%	High accuracy but too large for mobile
4th	MobileNetV3	5.83	96.72%	Excellent trade-off for entry-level phones
5th	AgriYouthNet	0.57	95.99%	Ultra-lightweight, purpose-built
💡 Critical Insights:
Context is King: Adding farmer-reported metadata (crop type, season, symptoms) to AgriYouthNet-Context achieves 99.06% accuracy—the highest overall—with only a 0.01MB size increase.

Purpose-Built Wins: Custom-designed AgriYouthNet (0.57MB) outperforms MobileNetV3 in efficiency, proving that architecture matters more than size.

Mobile-Ready: MobileNetV3 (5.83MB, 96.72%) is immediately deployable on entry-level phones (1-2GB RAM).

Quantization Works: INT8 quantization reduces model size by 75% with only 2.0% accuracy loss—ideal for extreme constraints.

📁 Repository Structure
text
AgriYouth-Plant-Disease-Detection/
│
├── 📓 **notebooks/**
│   └── 01_agriYouth_model_training_evaluation.ipynb   # Main experiment notebook
│
├── 📁 **outputs/**                      
│   └── figures/         # Learning curves, confusion matrices, charts
│
├── 📁 **reports/**
│   ├── final_report.pdf                                # Scholarly report
│   └── figures/          # Report figures
│
├── 📁 **docs/**
│   └── video_link.txt                                   # Link to demo video
│
├── 📄 **README.md**                                      #
├── 📄 **LICENSE**                                        # MIT License
├── 📄 **.gitignore**                                     # Files to exclude from git
└── 📄 **requirements.txt**                               # Python dependencies
🚀 Setup Instructions
Prerequisites
Python 3.12+

Kaggle account (for dataset download)

Google Colab account (recommended for GPU access)

Option 1: Run in Google Colab (Recommended)
https://colab.research.google.com/assets/colab-badge.svg

Simply click the badge above, and the notebook will open in Colab with all dependencies pre-installed.

Option 2: Local Installation
bash
# Clone the repository
git clone https://github.com/dahamkakooza/AgriYouth-Plant-Disease-Detection.git
cd AgriYouth-Plant-Disease-Detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
Option 3: Run All Experiments via Script
bash
# Coming soon - training pipeline
📊 Dataset
PlantVillage Dataset
The PlantVillage dataset contains over 50,000 images of healthy and diseased plant leaves across 38 classes.

Dataset Features:

Source: Kaggle (originally from Hughes & Salathé, 2015)

Size: ~650MB compressed, ~3GB extracted

Classes: 38 crop-disease pairs (e.g., "Tomato Early Blight")

Our subset: 10 classes, 12,206 images

Class Distribution (Selected 10 Classes):

Class	Crop	Disease	Images
0	Apple	Apple scab	~630
1	Apple	Black rot	~621
2	Apple	Cedar apple rust	~275
3	Blueberry	Healthy	~1,502
4	Cherry	Powdery mildew	~1,052
5	Corn	Cercospora leaf spot	~513
6	Corn	Common rust	~1,190
7	Corn	Northern Leaf Blight	~985
8	Grape	Black rot	~1,180
9	Grape	Esca (Black Measles)	~1,383
Data Augmentation
The notebook implements an AfricanAugmentationPipeline with:

Random brightness/contrast (simulating varying light)

Random fog/rain (simulating weather conditions)

Motion blur/ISO noise (simulating camera shake)

Image compression (simulating network constraints)

Random flips/rotations (augmenting training data)

🧠 Model Architectures
1. ResNet-50 (Baseline)
Parameters: 23.5 million

Size: 89.75 MB

Purpose: High-accuracy benchmark

Citation: He et al., 2016, "Deep Residual Learning for Image Recognition"

2. MobileNetV3 (Mobile-Optimized)
Parameters: 1.53 million

Size: 5.83 MB

Purpose: Efficient mobile deployment

Citation: Howard et al., 2019, "Searching for MobileNetV3"

3. EfficientNet-B0 (Balanced)
Parameters: 4.02 million

Size: 15.34 MB

Purpose: Best accuracy-size trade-off

Citation: Tan & Le, 2019, "EfficientNet: Rethinking Model Scaling"

4. AgriYouthNet (Custom Lightweight)
Parameters: 0.15 million

Size: 0.57 MB

Purpose: Ultra-efficient, purpose-built

Architecture: Custom CNN with depthwise separable convolutions

5. AgriYouthNet-Context (Context-Aware)
Parameters: 0.15 million

Size: 0.58 MB

Purpose: Leverage farmer-provided metadata

Context Features: Crop type, season, symptom pattern, region

📈 Experiments & Results
Experiment Summary Table
Exp ID	Model	Params (M)	Size (MB)	Val Acc (%)	Best Epoch	Time (min)	Mobile Ready?	Key Insight
E1	ResNet-50	23.53	89.75	97.75	6	20.0	❌ No	High accuracy, but too large for mobile
E2	MobileNetV3	1.53	5.83	96.72	9	4.3	✅ Yes	Excellent trade-off for entry-level phones
E3	EfficientNet-B0	4.02	15.34	98.48	3	12.1	⚠️ Mid-range	Best base accuracy, moderate size
E4	AgriYouthNet	0.15	0.57	95.99	8	3.8	✅ Yes	Ultra-lightweight custom design
E5	AgriYouthNet-Context	0.15	0.58	99.06	7	4.0	✅ Yes	🏆 BEST - Context awareness + efficiency
Visual Results
All visualizations are automatically saved in outputs/figures/:

Figure	Description	File
Learning Curves	Loss and accuracy over epochs	01_learning_curves.png
Confusion Matrices	Absolute and normalized	02_confusion_matrix.png
Error Analysis	Misclassification heatmaps	04_error_analysis.png
Model Comparison	Accuracy vs Size scatter plot	03_model_comparison.png
Experiment Summary	Summary cards per experiment	06_summary.png
Sample Visualization Outputs:
text
outputs/figures/
├── experiment1_resnet50/
│   ├── 01_learning_curves.png
│   ├── 02_confusion_matrix.png
│   └── 06_summary.png
├── experiment2_mobilenetv3/
│   ├── 01_learning_curves.png
│   ├── 02_confusion_matrix.png
│   └── 04_summary.png
├── experiment3_efficientnet/
│   ├── 01_learning_curves.png
│   ├── 02_confusion_matrix.png
│   └── 05_summary.png
└── experiment4_agriyouthnet/
    └── ...
📱 Mobile Deployment Considerations
Target Devices
Device	RAM	Processor	Suitability
Tecno Spark	1GB	MediaTek	Entry-level
Samsung A01	2GB	Qualcomm	Entry-level
Itel A56	1.5GB	Unisoc	Entry-level
Infinix Smart	2GB	MediaTek	Entry-level
Model Recommendations
User Segment	Recommended Model	Size	Accuracy	Rationale
Entry-level (1GB RAM)	AgriYouthNet	0.57 MB	95.99%	Fits in cache, fast inference
Entry-level (2GB RAM)	MobileNetV3	5.83 MB	96.72%	Better accuracy, still fits
Mid-range (3GB+ RAM)	EfficientNet-B0	15.34 MB	98.48%	Best accuracy for capable devices
All devices (with context)	AgriYouthNet-Context	0.58 MB	99.06%	🏆 Optimal choice
Optimization Techniques
INT8 Quantization (Implemented)

Reduces model size by 75% (0.57MB → 0.14MB)

Minimal accuracy loss (95.99% → 97.08%)

Pruning (Future work)

Remove redundant connections

Knowledge Distillation (Future work)

Train smaller student models from larger teachers

🔄 Reproducibility
All experiments are fully reproducible with:

Fixed Random Seeds
python
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
tf.random.set_seed(42)
Deterministic CUDA Operations
python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
Experiment Tracking
The notebook includes a custom ContextAwareExperimentTracker that logs:

Hyperparameters for each run

Training/validation metrics

Model sizes and training times

Insights and error analysis

To Reproduce Results:
Run the notebook from top to bottom (Cell → Run All)

All outputs will be saved in outputs/ directory

Experiment results are logged in outputs/results/experiment_table.csv

📄 Report & Demo
Final Report
The scholarly report is available at: reports/final_report.pdf

The report follows IEEE format and includes:

Problem definition with scholarly justification

Literature review with 10+ academic sources

Detailed methodology

Results with visualizations

Critical error analysis

Discussion and conclusions

Demo Video
The 5-10 minute presentation video can be accessed at:
Link in docs/video_link.txt

The video covers:

Problem statement and mission alignment

Dataset and preprocessing

Model architectures and experiments

Key results and findings

Mobile deployment considerations

📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

👤 Author
Kakooza Mahad

Course: Introduction to Machine Learning

Date: February 21, 2026

Email: k.mahad@alustudent.com

GitHub: @your-username

🙏 Acknowledgments
Dataset: PlantVillage by Hughes & Salathé, 2015

Kaggle: For hosting the dataset

PyTorch & TensorFlow Teams: For excellent deep learning frameworks

Google Colab: For free GPU access