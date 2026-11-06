# Domain Adaptation: from Public Fingertip PPG to Behind-the-Ear MAX30102

**Bottom line:** nothing in this repository shows that the models work on behind-ear PPG.
Every model was trained and tested on fingertip-type recordings. Behind-ear performance is
**unknown** until RakshPulse collects its own reference-labelled recordings.

## 1. How behind-ear PPG is expected to differ

| Factor | Fingertip (training data) | Behind ear / posterior auricular (target) | Expected effect on a fingertip-trained model |
|---|---|---|---|
| Vascular bed | Dense arteriolar/capillary bed, arteriovenous anastomoses, strong sympathetic vasomotor control | Thin skin over the mastoid/auricle, supplied by the posterior auricular artery (an external-carotid branch) | Different pulse shape. Ear-site pulses are closer to the central (carotid) waveform |
| Distance from heart | Long (aorta → brachial → radial → digital) | Short (carotid → external carotid) | Earlier arrival, smaller wave-reflection augmentation, so systolic/diastolic timing features shift |
| AC amplitude | Large | About 2–3× smaller (in-ear PPG vs finger: mean 2.35× lower, Davies et al. 2020) | Lower SNR. More windows rejected by the SQI |
| Vasoconstriction (cold, high altitude, stress) | Strong: finger perfusion collapses in cold | Ear/head perfusion is better preserved centrally | Possibly an *advantage* in the field, but it changes the feature distribution |
| Contact pressure | Clip or finger pad | Band/clip over bone. Pressure varies with head movement and helmet | Strong morphology changes (CP-PPG and WF-PPG datasets show contact pressure reshapes the waveform) |
| Motion | Hand motion | Jaw (talking, chewing), head movement, walking impacts | Artefact spectra differ from wrist/finger. No accelerometer on the MAX30102 |
| Measurement mode | Guilin: unspecified. Foggia: reflective. PPG-BP: clip (likely transmissive) | Reflective (MAX30102) | Reflective PPG samples shallower vessels |
| Hair, sweat, skin tone | Glabrous skin | Hair margin, sweat under helmets | Reduced and variable optical coupling |
| Wavelength | Red 660 / IR 880–940 | Same LEDs | Only factor that transfers directly |

Supporting observation from this repository: the IR ensemble beats of Guilin, Foggia and
PPG-BP (`docs/figures/mean_beat_by_dataset.png`) differ visibly between fingertip cohorts of
different age. Site differences are expected to be at least as large.

## 1b. Measured evidence: finger vs earlobe in the same people (Auricle)

The Auricle dataset (PhysioNet, 41 healthy subjects analysed) records finger and **earlobe** PPG
simultaneously with continuous CNAP BP. Earlobe is not posterior-auricular, and Nonin transmissive clips are not a
reflective MAX30102, but it is the closest open evidence available. Full tables are in MODEL_EVALUATION.md §F.

- **Morphology differs** (`docs/figures/auricle_ear_vs_finger_beat.png`): the earlobe pulse has a slower
  upstroke, a later systolic peak (~23 % vs ~15 % of the beat) and no diastolic shoulder. This matches the
  expectation in §1 of a smoother, more proximal waveform.
- **Signal quality was lower at the ear** (SQI pass 78.8 % vs 85.7 %; exercise 52.5 % vs 64.7 %), so the
  ear is not automatically the more robust site.
- **Finger-trained models do not transfer to the ear:** within-session SBP tracking fell from r 0.47
  (finger→finger) to r 0.08–0.29 (finger→ear). Unsupervised per-site standardisation removed the bias
  (−3.8 → +0.9 mmHg) but did **not** restore tracking.
- **Ear-trained models do work modestly:** ear→ear tracking r 0.39 (vs 0.21 for heart rate alone). With
  one-point calibration, SBP MAE was 14.10 vs 15.17 mmHg for carry-forward (ΔMAE −1.14, 95 % CI [−2.13, −0.18]).
  There was no DBP benefit.

**Consequence for RakshPulse:** a behind-ear model must be trained on behind-ear data with a calibration
step. Reusing fingertip models, even with feature re-scaling, is not supported by the evidence.

## 2. What the public-data work can and cannot tell us

**Can:**
- Whether the preprocessing, SQI gate and inference interface work end to end on real
  red/IR MAX3010x-class signals (they do).
- Which feature families carry *any* BP information in a young-adult cohort, and whether
  red + IR helps compared with IR alone (see MODEL_EVALUATION.md).
- How much of the apparent skill comes from demographics rather than the pulse wave.
- How badly a model degrades under a dataset shift (Guilin → Foggia/PPG-BP), as a lower
  bound on what a site shift may do.

**Cannot:**
- Give any accuracy figure for behind-ear placement.
- Validate the SQI thresholds for the ear site (amplitude and noise differ).
- Establish calibration-free population accuracy for soldiers in the field (altitude, cold,
  exertion, dehydration).

## 3. Adaptation strategies (to apply once behind-ear data exist)

1. **Feature-level alignment (first choice, small data):** use amplitude- and
   timing-normalised features (already done), and re-standardise features with ear-site
   statistics (domain-specific scaling). Cheap and testable with tens of subjects.
2. **Personal calibration (most realistic route to useful accuracy):** one or more
   cuff readings per user. The model then predicts *changes* from that user's baseline.
   Every validated cuffless device on the market today uses initial cuff calibration. ISO
   81060-3 and IEEE 1708 require evaluating calibration drift over time.
3. **Fine-tuning / transfer learning:** pre-train the CNN on fingertip data, then fine-tune
   the last layers on ear data. This only becomes meaningful with roughly 50+ ear subjects,
   given that even the 148-subject fingertip CNN did not beat the baseline here.
4. **Unsupervised domain adaptation** (e.g. CORAL, domain-adversarial training) using
   *unlabelled* behind-ear recordings. Labelled data are expensive, but unlabelled field
   data are cheap. This needs a labelled ear test set to verify any gain.
5. **Paired-site study (recommended next experiment):** (the Auricle analysis above is the public-data
   version of this, at the earlobe.) record finger PPG and behind-ear
   MAX30102 PPG at the same time, plus a validated cuff, on the same subjects. That allows
   direct finger↔ear morphology mapping and a fair test of transfer. The Auricle dataset
   (simultaneous finger + earlobe PPG with CNAP BP, PhysioNet) allows a preliminary
   finger→earlobe transfer study without new hardware, and should be the next download.

## 4. Minimum data-collection protocol suggested for the hardware team

- MAX30102 behind the ear: record red and IR at ≥ 100 sps (200 sps preferred), with raw
  FIFO counts (no on-chip averaging beyond what is logged), LED currents and pulse width logged,
  plus die temperature.
- Simultaneous finger PPG (any pulse oximeter with waveform output) for paired comparison.
- Reference: a validated automated upper-arm cuff (on a STRIDE BP or similar validated list),
  measured simultaneously on the opposite arm, or with sequential same-arm readings following
  ISO 81060-2 procedures.
- Conditions: seated rest, standing, after light exercise, talking/chewing, and walking, to
  test the SQI. Label each segment.
- Population: aim for a BP spread that includes hypertensive subjects. A narrow-BP cohort (like
  Guilin) cannot show that a model tracks BP.
- Ethics approval and informed consent before any data collection.
