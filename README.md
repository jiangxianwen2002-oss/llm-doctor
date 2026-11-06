# What Do LLM Doctors Add to Their Statistics? Dissecting Statistical Advice in Interactive Diagnosis

Code, prompts, run settings, and a statistical note for the paper.

## Contents

```
robustness_note.pdf         every comparison the paper draws a conclusion from: differences, 95% CIs, p, minimum detectable effects,
                            and the same without the patients whose condition has no labeled patients
setup_details.md            run settings and details not in the paper
t1_premise/                 statistician, advice, wrong advice, and interview code
  ddx_signal.py             statistician and advice line (posterior, expected information gain, stopping rule)
  t1_run.py, t1_run_s2.py   interview loop and prompts (no advice)
  t1_run_sig.py             interview with advice
  ddx_bad*.py, t1_run_bad*.py   wrong advice (RQ3): wrong diagnosis (dxonly), wrong questions (qonly), all wrong (contra)
  t1_run_budget.py          sets the question budget B
  test97.py                 offline tests of the wrong advice
  r96/t1_run_api.py         DS-V4.1-Flash API client (same prompts)
  r98/t1_run_api_qwen.py    Qwen3.8-Flash API client (same prompts)
  r99/t1_run_api_glm.py     GLM-5.3-Flash API client (same prompts)
  r96/run96.sh, r97/scripts/, r98/run98.sh, r99/run99.sh   launchers
training/train_b2s.py       LoRA SFT
tools/plan_a/               analysis
  verify_paper.py           recomputes every number in the paper's tables, figures and text from the interview logs
  implications.py           numbers of Section 4 (questions asked, disagreement with the statistician, model calls)
  robust_v1.py              all comparisons in robustness_note.pdf; make_robust_v1.py typesets the note
tools/dir34_menu/           data construction (make_*.py, build_ddx_signal.py)
```

Internal names in the code: "signal" = advice, "teacher" = statistician, "readout" = the statistician's most likely condition given the answers the LLM collected, "same-length" = the statistician asking as many questions as the LLM, "flash" = the larger models, UNTRAINED / NONE = off-the-shelf without advice, SIG-0 = off-the-shelf with advice, B2-R = SFT w/o advice, SIG-T = SFT w/ advice.

## Settings

Requirements: Python 3.9+ with `numpy` (analysis); `torch`, `transformers`, `peft` (training); an OpenAI-compatible server such as vLLM (local models).

Statistics: per-patient mean accuracy; patient-paired bootstrap, 10,000 resamples, seed 0, 95% intervals.

Interviews: temperature 0.8, at most 40 output tokens, 8 samples per patient (seeds 1-8) for Qwen 4B and Gemma 4B, 2 for each larger model; see `setup_details.md` for thinking settings.

Local models, for example Qwen 4B without advice at B=10:

```bash
python t1_run_budget.py 10 t1_run_s2.py --model <served model name> --out q_none_b10 --samples 8
```

Larger models (the API key is passed on stdin, never in argv or logs):

```bash
QWEN_API_KEY=... bash t1_premise/r98/run98.sh full
GLM_API_KEY=... bash t1_premise/r99/run99.sh full
```

Training (rank 16, alpha 32, learning rate 1e-4):

```bash
python training/train_b2s.py --model <path>/Qwen3.5-4B --data t1_premise/sig86/sft_sig.jsonl --out <path>/sft_w_advice --epochs 1 --seed 1
```

## Notes

- **Data.** Patients come from DDXPlus (Fansi Tchango et al., NeurIPS 2022 Datasets and Benchmarks). Data files, interview logs, result files, and fine-tuned weights are not included, so the scripts cannot be run from this repository alone; `tools/dir34_menu/make_*.py` and `build_ddx_signal.py` record how the data were built from the DDXPlus release.
- **Launchers** run from a folder that also holds the interview scripts (`t1_run*.py`, `ddx_*.py`), `data/` and `sig86/`.
- **Anonymization.** Comments, usage strings and launchers in a few scripts were edited to remove local paths and internal log names; the code logic is unchanged.
