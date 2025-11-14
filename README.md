# Multi-Agent Post-Processing Pipeline for Non-Native English ASR

**CSCI 5541 - Natural Language Processing | Fall 2024**  
**University of Minnesota**

**Project Website:** https://rishabh.my/multiagentteam-shuyu-drew/

---

## Team Members

- **Rishabh Agarwal** - [agarw266@umn.edu](mailto:agarw266@umn.edu)
- **Ella Boytim** - [boyti003@umn.edu](mailto:boyti003@umn.edu)
- **Sharon Soedarto** - [soeda002@umn.edu](mailto:soeda002@umn.edu)

**Team Mentors:** Shuyu Gan, Drew Gjerstad  
**Faculty Advisor:** Prof. Dongyang Kang

---

## Project Overview

Despite advances in automatic speech recognition (ASR), large models like Whisper exhibit higher Word Error Rates (WER) for non-native English speakers. This project proposes a **multi-agent post-processing pipeline** that improves ASR accuracy for non-native speakers **without retraining** the underlying model.

### Research Question

> Can modular post-processing agents, operating on ASR outputs, reduce the accuracy gap between native and non-native English speakers without retraining the underlying model?

---

## System Architecture

Our pipeline consists of three specialized agents:

```
┌─────────────────────┐
│   ASR Agent         │  OpenAI Whisper (small.en)
│   (Whisper)         │  → Baseline transcription
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Error Analysis      │  BERT token classifier
│ Agent               │  → 8 error types detection
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Correction Agent    │  Rule-based + confusion matrix
│                     │  → Targeted error fixes
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Evaluation Agent    │  WER, ΔWER, feedback loop
│                     │  → Quality assessment
└─────────────────────┘
```

### Key Components

1. **Error Analysis Agent (BERT-based)**
   - Token-level error classification
   - 8 error types: equal, substitution, accent_pronunciation, homophone, deletion, insertion, repetition, filler
   - Trained with inverse class weighting for imbalance

2. **Correction Agent**
   - Uses BERT error labels + phoneme confusion matrices
   - Applies minimal, targeted corrections
   - Confidence-threshold gating (only high-confidence fixes)

3. **Evaluation Agent**
   - Computes WER, CER, ΔWER (fairness metric)
   - Tracks correction quality (helpful/harmful/neutral)
   - Triggers feedback loop for iterative refinement

---

## Current Status (Midterm Checkpoint)

### ✅ Completed

- [x] Dataset preparation (100 L2-ARCTIC samples)
- [x] Baseline ASR with Whisper (WER: 9.6%, CER: 4.8%)
- [x] Manual error annotation (902 tokens labeled)
- [x] BERT error classifier trained (proof-of-concept)
- [x] Phoneme confusion matrix extraction (accent-specific patterns)
- [x] Error taxonomy validated (8 error types)

### 🔄 In Progress

- [ ] Full agent integration (BERT → Correction → Evaluation)
- [ ] Correction Agent implementation (rule-based core complete)
- [ ] Evaluation Agent feedback loop
- [ ] End-to-end pipeline testing

### 📋 Planned (By Final Submission)

- [ ] Expand dataset to 500-1000 samples (LearnerVoice, Common Voice)
- [ ] Retrain BERT classifier on larger data
- [ ] Comprehensive evaluation (4 approaches: Baseline, Rule-based, BERT, Hybrid)
- [ ] Ablation studies
- [ ] Final report and presentation

---

## Key Results (Pilot Study: n=100)

### Baseline ASR Performance

| Metric | Value |
|--------|-------|
| Overall WER | 9.6% |
| Overall CER | 4.8% |
| Average WER | 10.3% |
| Average CER | 5.4% |
| Max sentence length | 13 words |

### Error Label Distribution

| Error Type | Count | Percentage |
|------------|-------|------------|
| `equal` (correct) | 817 | 90.1% |
| `substitution` | 45 | 5.0% |
| `accent_pronunciation` | 17 | 1.9% |
| `homophone` | 10 | 1.1% |
| `deletion+equal` | 7 | 0.8% |
| `insertion` | 3 | 0.3% |
| Other | 3 | 0.3% |

**Class Imbalance Ratio:** 817:1 (extreme)

### BERT Error Classifier (Pilot)

| Metric | Value |
|--------|-------|
| Training Loss | 0.77 |
| Evaluation Loss | 3.01 |
| Training Samples | 70 |
| Validation Samples | 10 |
| Test Samples | 20 |

**Status:** Overfitting observed (expected with n=100). Model validates architecture but requires more data for production use.

### Phoneme Confusion Patterns (NCC Speaker)

```
Key Confusions (P > 0.05):
- R → L   (0.059)  # Classic r/l confusion
- R → AA1 (0.059)  # Vowelization of /r/
- N → D   (0.059)  # Nasal-stop confusion
- T → final_stop_deletion (0.059)  # Final consonant deletion
```

---

## Repository Structure

```
.
├── index.html              # Project website
├── files/                  # Website assets
│   ├── bulma.min.css
│   ├── styles.css
│   ├── pipeline_diagram.png
│   ├── error_distribution.png
│   └── project_proposal.pdf
├── notebooks/              # Colab notebooks
│   ├── 01_data_preparation.ipynb
│   ├── 02_baseline_asr.ipynb
│   ├── 03_error_agent_training.ipynb
│   ├── 04_correction_agent.ipynb (in progress)
│   └── 05_evaluation_pipeline.ipynb (in progress)
├── data/                   # Datasets
│   ├── L2_ARCTIC_subset/
│   │   ├── audio/
│   │   └── transcripts.csv
│   ├── l2_artic_results_final_df_annotated.csv
│   └── l2_artic_results_ncc_phoneme_confusion_matrix.csv
├── models/                 # Trained models
│   └── bert_error_agent_v1/
├── outputs/                # Experimental results
│   ├── asr_outputs_L2.json
│   ├── error_analysis_results.json
│   └── confusion_matrices.json
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

---

## Installation & Usage

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- GPU recommended (Google Colab T4 used for development)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/multiagent-asr.git
cd multiagent-asr

# Install dependencies
pip install -r requirements.txt
```

### Running the Pipeline (Coming Soon)

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

**Note:** Full pipeline integration is in progress. Individual notebooks are available in `notebooks/`.

---

## Datasets

### L2-ARCTIC (Current)
- **Source:** [Zhao et al., 2018](https://psi.engr.tamu.edu/l2-arctic-corpus/)
- **Samples Used:** 100 utterances (scripted speech)
- **Speakers:** NCC (Cantonese), AHW (Arabic)
- **License:** Open access

### LearnerVoice (Planned)
- **Source:** [Kim et al., 2024](https://arxiv.org/abs/2407.04280)
- **Content:** 50 hours spontaneous L2 English (L1 Korean)
- **Features:** Disfluency annotations
- **Status:** Access pending

### Mozilla Common Voice 15.0 (Planned)
- **Source:** [Mozilla Foundation](https://commonvoice.mozilla.org/)
- **Content:** 100+ accents, thousands of speakers
- **Use Case:** Diverse accent evaluation
- **Status:** Access pending

---

## Methodology

### 1. Error Annotation Process

Each ASR hypothesis word is manually labeled:
- Compare ASR output vs. ground truth
- Identify error type using our 8-class taxonomy
- Handle deletions by concatenating labels (e.g., `deletion+equal`)
- Extract phoneme-level confusions using forced alignment

### 2. BERT Token Classification

```python
# Simplified training code
model = AutoModelForTokenClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(unique_error_labels),
    id2label=id2label,
    label2id=label2id
)

# Custom trainer with class weights
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_loss=weighted_cross_entropy_loss
)
```

### 3. Correction Rules

Rules are guided by confusion matrices:
```python
if error_label == "accent_pronunciation":
    if P(hyp|ref, accent) > threshold:
        apply_phoneme_correction(token, confusion_matrix)
```

### 4. Evaluation Metrics

- **WER (Word Error Rate):** Primary accuracy metric
- **ΔWER (Delta WER):** Fairness metric = max(WER) - min(WER) across accents
- **Correction Quality:** % helpful vs. harmful vs. neutral corrections
- **Per-accent breakdown:** Identify which accents benefit most

---

## Expected Final Results

Based on related work (LearnerVoice fine-tuning reduced WER by 44%), we anticipate:

| Metric | Expected Improvement |
|--------|----------------------|
| WER Reduction | 15-30% relative |
| ΔWER Reduction | 20-40% |
| Correction Quality | >70% helpful, <10% harmful |
| Latency | <100ms per utterance |

---

## Limitations & Future Work

### Current Limitations

1. **Small training set (n=100)** → Severe overfitting in BERT classifier
2. **Extreme class imbalance (817:1)** → Rare error types poorly learned
3. **Limited accent coverage** → Only 2 L1 backgrounds (Cantonese, Arabic)
4. **Scripted speech only** → Fewer disfluencies than spontaneous speech

### Planned Improvements

1. **Scale to 500-1000 samples** with LearnerVoice and Common Voice
2. **Try focal loss** or data augmentation for class imbalance
3. **Test DistilBERT** for faster inference with similar accuracy
4. **Add more accents** (Spanish, Mandarin, Hindi, etc.)
5. **Evaluate on spontaneous speech** with disfluency handling

### Long-Term Vision

- Real-time correction API for voice interfaces
- Multi-lingual extension (non-English ASR)
- Integration with commercial ASR systems (Google, AWS, Azure)
- Personalized correction models per user/accent

---

## Novel Contributions

1. **Multi-Agent Architecture:** First post-ASR correction system with specialized agents and feedback loops
2. **Learned + Rule-Based Hybrid:** BERT error detection + phoneme confusion guidance
3. **Comprehensive Error Taxonomy:** 8-class system covering accent-specific and disfluency errors
4. **Fairness-Focused Evaluation:** Explicit ΔWER measurement for equity
5. **Model-Agnostic Design:** Works with any ASR system without retraining

---

## References

1. Radford et al. (2023). "Robust Speech Recognition via Large-Scale Weak Supervision." *OpenAI Whisper Technical Report.*
2. Kim et al. (2024). "LearnerVoice: A Dataset of Non-Native English Learners' Spontaneous Speech." *EMNLP 2024.*
3. Zhao et al. (2018). "L2-ARCTIC: A Non-Native English Speech Corpus." *Interspeech 2018.*
4. Koenecke et al. (2020). "Racial Disparities in Automated Speech Recognition." *PNAS.*
5. Feng et al. (2021). "Quantifying Bias in Automatic Speech Recognition." *Interspeech 2021.*
6. Mozilla Foundation (2024). "Mozilla Common Voice 15.0." *HuggingFace Datasets.*

---

## Citation

If you use this work, please cite:

```bibtex
@misc{multiagent-asr-2024,
  author = {Agarwal, Rishabh and Boytim, Ella and Soedarto, Sharon},
  title = {Multi-Agent Post-Processing Pipeline for Non-Native English ASR},
  year = {2024},
  publisher = {University of Minnesota},
  howpublished = {\url{https://github.com/yourusername/multiagent-asr}}
}
```

---

## Acknowledgments

We thank Professor Dongyang Kang and TAs Shuyu Gan and Drew Gjerstad for their guidance. We acknowledge the creators of L2-ARCTIC, LearnerVoice, and Mozilla Common Voice for making their datasets publicly available. Computational resources provided by Google Colab.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

For questions or collaborations:
- **Rishabh Agarwal:** agarw266@umn.edu
- **Ella Boytim:** boyti003@umn.edu
- **Sharon Soedarto:** soeda002@umn.edu

---

**Last Updated:** November 14, 2024  
**Status:** Midterm Checkpoint | Final submission December 2024