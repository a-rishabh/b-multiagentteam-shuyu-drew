# Multi-Agent Post-Processing Pipeline for Non-Native English ASR

**CSCI 5541 - Natural Language Processing | Fall 2024**  
**University of Minnesota**

**Project Website:** https://rishabh.my/multiagentteam-shuyu-drew/

---

## Team Members

- **Rishabh Agarwal** - agarw266@umn.edu
- **Ella Boytim** - boyti003@umn.edu
- **Sharon Soedarto** - soeda002@umn.edu

**Team Mentors:** Shuyu Gan, Drew Gjerstad  
**Faculty Advisor:** Prof. Dongyang Kang

---

## 🎯 Project Overview

Despite advances in automatic speech recognition (ASR), large models like Whisper exhibit higher Word Error Rates (WER) for non-native English speakers. This project proposes a **multi-agent post-processing pipeline** that improves ASR accuracy for non-native speakers **without retraining** the underlying model.

### Research Question

> Can modular post-processing agents, operating on ASR outputs, reduce the accuracy gap between native and non-native English speakers without retraining the underlying model?

---

## 🏗️ System Architecture

Our pipeline consists of three specialized agents:

```
┌───────────────────────┐
│   ASR Agent           │  OpenAI Whisper (small.en)
│   (Whisper)           │  → Baseline transcription
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│ Error Analysis        │  BERT token classifier
│ Agent                 │  → 8 error types detection
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│ Correction Agent      │  Rule-based + confusion matrix
│                       │  → Targeted error fixes
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│ Evaluation Agent      │  WER, ΔWER, feedback loop
│                       │  → Quality assessment
└───────────────────────┘
```

### Key Components

1. **Error Analysis Agent (BERT-based)**
   - Token-level error classification
   - 8 error types: equal, substitution, accent_pronunciation, homophone, deletion, insertion, repetition, filler
   - Trained with inverse class weighting for extreme imbalance (817:1 ratio)

2. **Correction Agent**
   - Uses BERT error labels + phoneme confusion matrices
   - Applies minimal, targeted corrections
   - Confidence-threshold gating (only high-confidence fixes)

3. **Evaluation Agent**
   - Computes WER, CER, ΔWER (fairness metric)
   - Tracks correction quality (helpful/harmful/neutral)
   - Triggers feedback loop for iterative refinement

---

## 📊 Current Results (Midterm Checkpoint - n=100)

### Baseline ASR Performance

| Metric | Value | Notes |
|--------|-------|-------|
| **Overall WER** | **9.6%** | Better than expected for non-native speech |
| **Overall CER** | **4.8%** | Character-level accuracy |
| **Average WER** | 10.3% | Per-utterance average |
| **Average CER** | 5.4% | Per-utterance average |
| **Max Sentence Length** | 13 words | Important for context window |
| **Total Tokens Annotated** | 902 | Word-level error labels |

### Error Label Distribution

| Error Type | Count | Percentage | Description |
|------------|-------|------------|-------------|
| `equal` (correct) | **817** | **90.1%** | ✓ Correct transcription (majority class) |
| `substitution` | 45 | 5.0% | Word replaced with different word |
| `accent_pronunciation` | 17 | 1.9% | Accent-driven phonetic error |
| `homophone` | 10 | 1.1% | Sound-alike substitution (e.g., "20th" vs "twentieth") |
| `deletion+equal` | 7 | 0.8% | Missing word (label concatenated with next) |
| `insertion` | **3** | **0.3%** | ❌ Extra word inserted (insufficient for learning) |
| Other combinations | 3 | 0.3% | Compound errors |

**Class Imbalance Ratio:** 817:1 (extreme)

### BERT Error Classifier Results

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Training Loss** | 0.77 | Low training error |
| **Eval Loss** | **3.01** | ❌ Severe overfitting |
| **Training Samples** | 70 | Insufficient for generalization |
| **Validation Samples** | 10 | |
| **Test Samples** | 20 | |
| **Overfitting Gap** | ~3.9x | Eval loss / Train loss ratio |

**Status:** ⚠️ Expected limitation with n=70. Model validates architecture but requires 500-1000 samples for production use.

### Phoneme Confusion Patterns (NCC Speaker)

| Reference Phoneme | Confused With | Probability | Linguistic Pattern |
|-------------------|---------------|-------------|-------------------|
| `R` | `L`, `AA1` | ~0.059 each | Classic r/l confusion + vowelization |
| `N` | `D`, `L` | ~0.059 each | Nasal-stop confusion |
| `T` | `D`, `final_stop_deletion` | ~0.059 each | Final consonant deletion (Cantonese) |
| `D` | `V`, `Z` | ~0.059 each | Stop-fricative confusion |
| Vowels | Multiple patterns | ~0.059 | AA1↔AH1, AY1↔IH1, EH1↔DH |

---

## ✅ Current Status (Midterm Checkpoint)

### Completed
- [x] Dataset preparation (100 L2-ARCTIC samples, 902 tokens annotated)
- [x] Baseline ASR with Whisper (WER: 9.6%, CER: 4.8%)
- [x] Manual error annotation with 8-class taxonomy
- [x] BERT error classifier trained (proof-of-concept, overfitting expected)
- [x] Phoneme confusion matrix extraction (accent-specific patterns)
- [x] Error taxonomy validated and tested

### 🔄 In Progress
- [ ] Full agent integration (BERT → Correction → Evaluation)
- [ ] Correction Agent implementation (rule-based core complete)
- [ ] Evaluation Agent feedback loop
- [ ] End-to-end pipeline testing on pilot data

### 📋 Planned (By Final Submission - December 2024)
- [ ] Expand dataset to 500-1000 samples (LearnerVoice, Common Voice)
- [ ] Retrain BERT classifier on larger data (target: eval loss < 1.0)
- [ ] Comprehensive evaluation (4 approaches: Baseline, Rule-based, BERT, Hybrid)
- [ ] Ablation studies (which agent contributes most?)
- [ ] Final report and presentation

---

## 🚀 Installation & Usage

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- GPU recommended (Google Colab T4/A100 used for development)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/multiagent-asr.git
cd multiagent-asr

# Install dependencies
pip install -r requirements.txt
```

### Running the Pipeline (Coming Soon - Integration in Progress)

```bash
# 1. Prepare data
python scripts/prepare_data.py --dataset l2arctic --samples 100

# 2. Run baseline ASR
python scripts/run_baseline.py --model whisper-small.en

# 3. Train error agent (or load pretrained)
python scripts/train_error_agent.py --data data/annotated.csv

# 4. Run full pipeline
python scripts/run_pipeline.py --input data/test.csv --output results/
```

**Note:** Full pipeline integration is currently in progress. Individual notebooks are available in `notebooks/`.

---

## 📁 Repository Structure

```
.
├── index.html                  # Project website
├── files/                      # Website assets
│   ├── bulma.min.css
│   ├── styles.css
│   ├── pipeline_diagram.png
│   ├── error_distribution.png
│   └── project_proposal.pdf
├── notebooks/                  # Colab notebooks
│   ├── 01_data_preparation.ipynb
│   ├── 02_baseline_asr.ipynb
│   ├── 03_error_agent_training.ipynb
│   ├── 04_correction_agent.ipynb (in progress)
│   └── 05_evaluation_pipeline.ipynb (in progress)
├── data/                       # Datasets
│   ├── L2_ARCTIC_subset/
│   │   ├── audio/
│   │   └── transcripts.csv
│   ├── l2_artic_results_final_df_annotated.csv  # 902 tokens with error labels
│   └── l2_artic_results_ncc_phoneme_confusion_matrix.csv  # Accent patterns
├── models/                     # Trained models
│   └── bert_error_agent_v1/    # Proof-of-concept BERT classifier
├── outputs/                    # Experimental results
│   ├── asr_outputs_L2.json
│   ├── error_analysis_results.json
│   └── confusion_matrices.json
├── requirements.txt            # Python dependencies
└── README.md                  # This file
```

---

## 📊 Datasets

### L2-ARCTIC (Current - n=100)
- **Source:** [Zhao et al., 2018](https://psi.engr.tamu.edu/l2-arctic-corpus/)
- **Samples Used:** 100 utterances (scripted speech)
- **Speakers:** NCC (Native Cantonese Chinese), AHW (Arabic)
- **License:** Open access
- **Status:** ✅ Complete

### LearnerVoice (Planned - n=500)
- **Source:** [Kim et al., 2024](https://arxiv.org/abs/2407.04280)
- **Content:** 50 hours spontaneous L2 English (L1 Korean)
- **Features:** Disfluency annotations
- **Status:** 🔄 Access pending

### Mozilla Common Voice 15.0 (Planned - n=500)
- **Source:** [Mozilla Foundation](https://commonvoice.mozilla.org/)
- **Content:** 100+ accents, thousands of speakers
- **Use Case:** Diverse accent evaluation
- **Status:** 🔄 Access pending

---

## 🔬 Methodology

### 1. Error Annotation Process

Each ASR hypothesis word is manually labeled:
1. Compare ASR output vs. ground truth
2. Identify error type using our 8-class taxonomy
3. Handle deletions by concatenating labels (e.g., `deletion+equal`)
4. Extract phoneme-level confusions using forced alignment

### 2. BERT Token Classification

```python
# Training configuration
model = AutoModelForTokenClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=9,  # 8 error types + padding
    id2label=id2label,
    label2id=label2id
)

# Custom trainer with inverse class weights
# Weights range: 1.0 (equal) to 817.0 (rarest classes)
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,  # n=70
    eval_dataset=val_dataset,      # n=10
    compute_loss=weighted_cross_entropy_loss
)

# Training: 8 epochs, LR=3e-5, batch_size=2
# Result: Train loss 0.77 → Eval loss 3.01 (overfitting expected)
```

### 3. Correction Rules

Rules guided by confusion matrices:
```python
if error_label == "accent_pronunciation":
    if P(hyp|ref, accent) > threshold:
        apply_phoneme_correction(token, confusion_matrix)
```

### 4. Evaluation Metrics

- **WER (Word Error Rate):** Primary accuracy metric
- **ΔWER (Delta WER):** Fairness metric = max(WER) - min(WER) across accents
- **Correction Quality:** % helpful vs. harmful vs. neutral
- **Per-accent breakdown:** Identify which accents benefit most

---

## 🎯 Expected Final Results (n=500-1000)

Based on related work (LearnerVoice fine-tuning reduced WER by 44%), we anticipate:

| Metric | Current (n=100) | Expected (n=1000) |
|--------|-----------------|-------------------|
| **WER Reduction** | TBD (integration pending) | **15-30%** relative |
| **ΔWER Reduction** | TBD | **20-40%** |
| **Correction Quality** | TBD | **>70%** helpful, **<10%** harmful |
| **Inference Latency** | TBD | **<100ms** per utterance |
| **BERT Eval Loss** | 3.01 (overfitting) | **<1.0** (generalizing) |

---

## ⚠️ Current Limitations & Mitigation

### 1. Small Training Set (n=70) → Severe Overfitting
- **Issue:** Train loss (0.77) vs. Eval loss (3.01) = 3.9x gap
- **Mitigation:** 
  - ✅ Validates architecture (proof-of-concept achieved)
  - 📋 Retrain with 500-1000 samples
  - 📋 Use DistilBERT (66M params) instead of BERT (110M)

### 2. Extreme Class Imbalance (817:1)
- **Issue:** 90% tokens are "equal", only 3 insertion examples
- **Mitigation:**
  - ✅ Inverse class weighting (implemented)
  - 📋 Focal loss (planned)
  - 📋 Synthetic data generation (planned)
  - 📋 Binary classification fallback (correct vs. error)

### 3. Limited Accent Coverage (2 L1 backgrounds)
- **Issue:** Only Cantonese and Arabic speakers
- **Mitigation:** 
  - 📋 Expand to 5+ accents via Common Voice 15.0

### 4. Scripted Speech Only
- **Issue:** L2-ARCTIC has few disfluencies (0.2%)
- **Mitigation:** 
  - 📋 Add LearnerVoice (spontaneous speech with disfluencies)

---

## 💡 Novel Contributions

1. **Multi-Agent Architecture:** First post-ASR correction system with specialized, communicating agents and feedback loops
2. **Learned + Rule-Based Hybrid:** BERT error detection + phoneme confusion guidance
3. **Comprehensive Error Taxonomy:** 8-class system covering accent-specific + disfluency errors
4. **Innovative Deletion Handling:** Concatenating labels (e.g., `deletion+equal`) avoids complex realignment
5. **Fairness-Focused Evaluation:** Explicit ΔWER measurement for equity
6. **Model-Agnostic Design:** Works with any ASR system without retraining

---

## 📚 Key References

1. Radford et al. (2023). "Robust Speech Recognition via Large-Scale Weak Supervision." *OpenAI Whisper Technical Report.* arXiv:2212.04356.
2. Kim et al. (2024). "LearnerVoice: A Dataset of Non-Native English Learners' Spontaneous Speech." *EMNLP 2024.* arXiv:2407.04280.
3. Zhao et al. (2018). "L2-ARCTIC: A Non-Native English Speech Corpus." *Interspeech 2018.*
4. Koenecke et al. (2020). "Racial Disparities in Automated Speech Recognition." *PNAS* 117(14):7684-7689.
5. Feng et al. (2021). "Quantifying Bias in Automatic Speech Recognition." *Interspeech 2021.*
6. Mozilla Foundation (2024). "Mozilla Common Voice 15.0." *HuggingFace Datasets.*

---

## 🙏 Acknowledgments

We thank Professor Dongyang Kang and TAs Shuyu Gan and Drew Gjerstad for their guidance. We acknowledge the creators of L2-ARCTIC, LearnerVoice, and Mozilla Common Voice for making their datasets publicly available. Computational resources provided by Google Colab.

---

## 📬 Contact

For questions or collaborations:
- **Rishabh Agarwal:** agarw266@umn.edu
- **Ella Boytim:** boyti003@umn.edu
- **Sharon Soedarto:** soeda002@umn.edu

---

**Last Updated:** November 14, 2024  
**Status:** Midterm Checkpoint Complete | Final submission December 2024