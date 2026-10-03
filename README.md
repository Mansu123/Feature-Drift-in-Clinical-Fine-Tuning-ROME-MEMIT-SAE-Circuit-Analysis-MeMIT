

# Clinical Feature Drift: Are "Local" Knowledge Edits Really Local?

Knowledge-editing methods such as ROME and MEMIT promise to change **one** fact inside a language model without touching anything else. In medicine, that promise matters: an edit that silently corrupts a neighbouring drug fact is a patient-safety problem.

This project edits a single clinical fact in **BioMistral-7B** and then measures what else moved, using a sparse autoencoder (SAE) trained on the unedited model, together with logit lens, activation patching, layer-wise CKA, and a downstream behavioural audit.

**Short answer: the edit was not local.** Global similarity metrics barely changed, but several *related* clinical facts were corrupted.

---

## Key results

**The edit itself.** One fact was changed with MEMIT:

| Prompt | Before edit | After edit |
|---|---|---|
| The maximum recommended daily dose of metformin is | 3 g | 5000 mg per day |

**Collateral damage on facts that were never edited:**

| Prompt | Before edit | After edit |
|---|---|---|
| Max daily dose of ibuprofen | 4000 mg | 600 mg |
| Max daily dose of acetaminophen | 4 g (adults) | 5000 mg per day |
| Primary mechanism of action of warfarin | Inhibition of vitamin K–dependent synthesis | "150 mg in 24 hours" (a dosage, not a mechanism) |
| First-line treatment for type 2 diabetes | Metformin | Regular insulin |

**Global metrics did not detect it:**

| Measure | Result |
|---|---|
| Layer-wise linear CKA (pre vs. post, all layers) | ≥ 0.9999 |
| SAE held-out reconstruction (fraction of variance unexplained) | 0.05 |
| Mean decoder overlap between edit features and drifted features | 0.037 |
| Correlation between decoder overlap and drift | −0.08 |
| Activation-patching recovery at the edit layer | −11.4% (edit not cleanly localized) |

The damage is invisible to representation-level similarity but obvious in behaviour. That gap is what makes edits like this dangerous in clinical settings.

---

## Pipeline

All phases live in one notebook and run top to bottom on a single Colab L4 GPU (22 GB).

| Phase | What it does |
|---|---|
| **1. Baseline SAE** | Collects 300K residual-stream activations at layer 15 from PubMedQA (`pqa_artificial`, 4,000 documents) and trains a Top-K SAE (32,768 features, k = 32). |
| **2. MEMIT edit** | Writes the new metformin dose into a band of four MLP `down_proj` layers (4–7), with a hard Frobenius-norm cap on each layer's update. |
| **3. Drift analysis** | Runs the frozen SAE on pre- and post-edit activations and ranks features by drift; inspects top-activating contexts; logit lens across layers; activation patching; layer-wise CKA. |
| **4. Conflict matrix** | Builds a decoder-overlap matrix between edit-related and drifted features and tests whether overlap predicts drift. |
| **5. Clinical harm audit** | Probes related drug and treatment facts before and after the edit, and evaluates on PubMedQA (`pqa_labeled`). |

### Setup

```bash
pip install -U transformers accelerate datasets matplotlib
```

Open the notebook in Colab with an L4 (or larger) GPU and run all cells in order. The model is loaded once and guarded against double-loading; the edit step restores the original weights before every run, so re-running cells is safe.

---

## Engineering challenges

**Memory.** The fp16 model alone uses about 14.5 GB of the L4's 22 GB, leaving little room for 300K activation vectors and a 32K-feature SAE. Out-of-memory errors were constant. Fixes:
- statistics are computed in a streaming fashion instead of materializing full feature matrices;
- after training, the optimizer is deleted and the SAE is moved to CPU;
- `expandable_segments` allocation and an explicit `free()` between phases.

**Edit stability.** The first approach, ROME, writes the whole fact into a single layer and repeatedly produced NaN or incoherent outputs. Because every later phase depends on the edited model, one bad edit silently invalidated everything downstream. Root causes found:
- an fp16 optimizer path that produced NaNs;
- re-running the edit cell stacked edits on already-modified weights;
- the target token resolving to whitespace rather than a number.

The fix was to switch to **MEMIT**, which spreads the update across four layers with a per-layer norm cap so no single layer can break; to save pristine weights at load time so the edit is idempotent; and to add a **sanity gate** that blocks all analysis unless logits are finite and the generation is coherent.

### Trade-offs

- **PubMedQA instead of MIMIC-III** for the SAE corpus, to avoid weeks of PhysioNet credentialing.
- **MEMIT over ROME:** weaker single-layer localization, in exchange for edits that are stable and trustworthy.
- **Single-GPU scale:** 4,000 training documents and 150 evaluation questions, rather than waiting for more compute.

### Development notes

The full pipeline was built and debugged in about one week, with roughly 10 hours of focused work alongside a full-time ML engineering role. Most of that time went into a tight debugging loop: run the notebook, read the traceback, find the root cause, rerun — since every failure in the edit step meant repeating all downstream analyses.

---

## Limitations

- **One edit.** All findings come from a single counterfactual. They show that collateral damage *can* happen, not how often or how badly it happens in general.
- **Small behavioural evaluation.** PubMedQA accuracy was 31.3% before and 28.7% after the edit on 150 questions. The baseline is near chance for this format, so this number should not be read as evidence on its own.
- **Localization is unresolved.** Activation patching did not recover the edited behaviour (−11.4%), so the mechanism behind the drift is not yet identified.
- **Decoder overlap did not predict drift** (r ≈ −0.08). Whatever links the edit to the corrupted facts is not captured by simple overlap between SAE decoder directions.
- **Known bug:** the "live features" diagnostic reports more than 100% and should be ignored until fixed.

---

## Future directions: roadmap to publication

**Target:** NeurIPS / ICLR / ACL / EMNLP main track, as a systematic study of **"Collateral Damage of Knowledge Editing in High-Stakes Domains."**

### 1. Scale the dataset (N ≥ 100–500 edits)
Build a dataset of clinical counterfactuals that spans entity types:
- drug dosages,
- contraindications,
- first-line treatments,
- mechanisms of action.

Pair each edit with a set of *related* and *unrelated* probe facts, so collateral damage can be measured by semantic distance from the edited fact, not just observed anecdotally.

### 2. Systematize the metrics
Measure off-target decay explicitly instead of through hand-picked probes:
- automated perplexity on held-out clinical text;
- downstream medical question answering (MedQA, PubMedQA) with a baseline well above chance;
- standard locality and specificity suites, adapted to clinical entities.

### 3. Formalize the SAE analysis
Use the SAE feature space to explain *why* the corruption happens, not only that it happens:
- does MEMIT shift specific **shared feature directions** that encode related drug knowledge?
- or does it distort **non-orthogonal features** in the MLP bottleneck through superposition?
- compare against ROME and fine-tuning baselines, and repeat activation patching at finer granularity to localize the effect.

---

## Repository

```
├── clinical_feature_drift_MEMIT.ipynb   # full pipeline (phases 1–5)
└── README.md
```

## Acknowledgements

- Model: [BioMistral-7B](https://huggingface.co/BioMistral/BioMistral-7B)
- Data: [PubMedQA](https://huggingface.co/datasets/qiaojin/PubMedQA)
- Editing methods: ROME (Meng et al., 2022) and MEMIT (Meng et al., 2023)

## Author

**Mansuba Tabassum** · [GitHub](https://github.com/Mansu123)
