# PPG Blood-Pressure Dataset Research for RakshPulse

*Research date: 2026-11-06. Every licence and access statement below was checked on the
hosting page that day, unless it is marked "not re-verified".*

## 1. Executive summary

- **No public dataset contains behind-the-ear PPG over the posterior auricular artery with
  reliable BP labels.** This is a **research gap**. RakshPulse will have to record its own
  behind-ear data with the MAX30102 against a reference cuff (see `DOMAIN_ADAPTATION.md`).
- **The best dataset for initial ML experiments is the Guilin dual-wavelength ECG+PPG dataset**
  (Mendeley `pz2zzr8vhm`, CC BY 4.0, 2026). It has raw (DC-preserving) **red and IR** PPG at
  250 Hz from a MAX3010x-class dual-LED front-end, 180 s at rest per subject, 148
  **healthy young adults aged 18–36** (close to a serving-soldier age profile), and a cuff SBP/DBP
  for every subject. Its weaknesses are a single cuff reading per subject, a narrow BP range,
  and a fingertip measurement site.
- **External validation:** the Foggia MAX86150 red+IR fingertip dataset (127 subjects aged
  18–86, CC BY 4.0) and PPG-BP (219 subjects, IR only, CC0).
- **The closest dataset to the eventual hardware is Auricle** (PhysioNet, ODC-By, 2026). It
  has transmissive **earlobe** PPG and finger PPG recorded simultaneously, with continuous CNAP
  BP and arm-cuff BP, in 70 subjects. It is the only open dataset found that pairs an ear-site
  PPG with BP labels. It is still not behind-ear reflective PPG, and at 15.4 GB it was not
  downloaded in this session (see section 13).
- Large ICU datasets (PulseDB, MIMIC-III, VitalDB, UCI) have many recordings, but they were
  ranked down. They provide **only a single fingertip PPG channel** (no red/IR separation) from
  ICU monitors, with **invasive arterial** BP labels in a critically ill population. That is a
  fundamentally different measurement problem from a field cuff-referenced spot check.
  Dataset size was **not** allowed to compensate for this.

## 2. Search methodology

- Sources searched: PhysioNet index (topic "ppg", "blood pressure"), Mendeley Data, figshare
  (including the CardioWorks team collection), Zenodo, UCI ML Repository, Kaggle/GitHub
  (PulseDB), Peter Charlton's curated PPG-dataset list, the ChatPPG dataset directory (used
  only for discovery; every fact was then checked on the primary host), PubMed/IOP/Nature
  dataset descriptors.
- Query families used: "PPG blood pressure dataset", "red infrared PPG blood pressure",
  "MAX30102/MAX30101/MAX86150 dataset", "ear / in-ear / behind-the-ear PPG blood pressure",
  "posterior auricular PPG", "cuffless blood pressure open dataset".
- Inclusion: publicly described dataset with PPG waveforms **and** SBP/DBP labels. Two
  ear-PPG datasets without BP (EarSet) were included only for the domain-adaptation analysis.
- Each candidate was then checked on its primary landing page for signals, sampling rate,
  BP reference method, site, licence and access conditions.

## 3. Dataset comparison table

| Dataset | Subj. | PPG channels | λ (nm) | fs (Hz) | Site | BP reference | Licence / access |
|---|---:|---|---|---:|---|---|---|
| **Guilin dual-wave ECG+PPG** (Mendeley pz2zzr8vhm v1, 2026) | 148 | Red + IR (raw ADC, DC kept) + lead-I ECG | 660/≈940 per a third-party index; *not stated by the authors* | 250 | Not stated in README (indexed as finger-contact) | 1 oscillometric cuff SBP/DBP per subject, at rest, temporally adjacent to recording 1 | CC BY 4.0, open download |
| **Foggia multiwavelength** (Mendeley gs7jvwtcgt v1, 2025) | 127 | IR + Red (pre-filtered, z-scored) | MAX86150 red/IR | 100 | Index fingertip, reflective | Mediland RC7F oscillometric cuff, contralateral arm, simultaneous | CC BY 4.0, open |
| **PPG-BP** (figshare 5459299 v5, 2018) | 219 | IR only (red not released) | 660/905 sensor; 905 IR released | 1000 | Fingertip (SEP9AF-2 clip) | Omron HEM-7201 cuff, before PPG | CC0, open |
| **Pulse Transit Time PPG** (PhysioNet, 2022) | 22 | Red, IR, green ×2 sites | MAX30101: 660/880/537 | 500 (CSV) | Finger, distal + proximal phalanx | Omron HEM-7322 cuff at start/end of each activity | ODbL 1.0, open (2.9 GB) |
| **Auricle** (PhysioNet, 2026) | 70 (42 healthy + 28 HF) | Finger + **earlobe** (Nonin, transmissive) | Nonin (not specified in excerpt) | 1000 | Finger, **earlobe** | Continuous CNAP finger BP + periodic arm cuff | ODC-By 1.0, open (15.4 GB) |
| **CardioWave-Portable** (PhysioNet, 2026) | 979 | 1 PPG + 1-lead ECG | Not stated | 500 | Finger (card device) | Cuff SBP/DBP | **Credentialed access** (DUA + training) |
| **PulseDB v2** (MIMIC-III + VitalDB) | 5,361 | 1 PPG (pleth) + ECG | Monitor-dependent, not reported | 125 | Fingertip (ICU / OR) | **Invasive arterial line** | MIMIC part ODbL; VitalDB part CC BY-NC-SA 4.0 (non-commercial) |
| **MIMIC-III Waveform Matched** | 10,282 pts (subset with PPG+ABP ≈ 2.8k) | 1 PPG | Not reported | 125 | Fingertip, ICU | Invasive ABP | ODbL, open |
| **MIMIC-IV Waveform v0.1** | ≈200 records | 1 PPG | Not reported | 62.5–125 | Fingertip, ICU | Invasive ABP | ODbL, open (pilot release) |
| **VitalDB** | 6,388 surgical pts | 1 PPG | Not reported | 100–500 | Fingertip, OR, anaesthetised | Invasive ABP + NIBP | **CC BY 4.0** on PhysioNet v1.0.0 (doi:10.13026/czw8-9p62), verified in 3.0 from the PhysioNet project page and LICENSE file. PulseDB redistributes a derived subset under CC BY-NC-SA. Used in 3.0 as an *external* test of "beyond the last real cuff reading" (data/README.md) |
| **UCI Cuff-Less BP** (MIMIC-II derived) | Unknown (no subject IDs) | 1 PPG | — | 125 | Fingertip, ICU | Invasive ABP | CC BY 4.0. **No subject IDs, so subject-wise splitting is impossible** |
| **BUT PPG v2** | 50 | Smartphone camera (RGB video-derived) | Phone flash | 30 | Finger **or ear (phone held to ear)** | Single Fora D40 cuff value per recording | CC BY 4.0, open |
| **CP-PPG** (figshare, 2025) | 142 | Dual-channel fingertip at several contact pressures | Not re-verified | — | Fingertip | Cuff SBP/DBP | figshare, open (details not re-verified) |
| **Four-wavelength PPG BP** (figshare 23283518) | — | 4 wavelengths, reflective | Not re-verified | — | Index finger | Omron HEM-7201, before PPG | figshare (details not re-verified) |
| **Aurora-BP** (Microsoft Research) | 1,125 | Wrist optical + tonometry + ECG | — | — | Wrist | Oscillometric + auscultatory cuff | Data-use agreement on request (*not re-verified*) |
| **EarSet** (Zenodo, 2023) | 30 | In-ear green/red/IR | green/red/IR | — | Ear canal | **No BP** | Open (Zenodo). Morphology reference only |

## 4. Signal specifications (shortlist, verified on downloaded data where available)

| | Guilin | Foggia | PPG-BP |
|---|---|---|---|
| Verified fs | 250 Hz (header `{'Hz': 250}`) | 100 Hz | 1000 Hz |
| Record length | ≈180 s × 3 states (45,425 samples checked for subject 001) | 10 s | 2.1 s × 3 |
| Resolution | Integer ADC counts around 4×10⁵, consistent with an 18-bit MAX3010x-class front end | float, z-scored | 12-bit ADC (paper) |
| DC preserved | **Yes** (raw folder) | No | No (hardware 0.5–12 Hz band-pass) |
| Missing values | None found in rest recordings | None | None |
| Motion | Rest (state 1). States 2–3 are post-exercise **without BP** | Seated rest | Rest |

## 5. BP label quality

| Dataset | Label type | Timing | Caveat |
|---|---|---|---|
| Guilin | Oscillometric cuff, 1 reading | Rest recording; exact cuff-to-PPG timing not specified | One label shared by all windows of a subject, so windows are *not* independent samples |
| Foggia | Oscillometric cuff (CE, ISO 13485 maker) | Simultaneous, contralateral arm | Single reading |
| PPG-BP | Omron HEM-7201 cuff | Measured *before* the PPG segments | Single reading |
| PulseDB / MIMIC / VitalDB / UCI | **Invasive radial ABP**, beat-level | Simultaneous | Not interchangeable with cuff BP: different site, ICU haemodynamics, sedation/vasopressors |
| Auricle | Continuous **CNAP** finger BP + arm cuff | Simultaneous | CNAP is itself calibrated, and the dataset masks calibration periods |

## 6. Subject population

| Dataset | Age | Sex | BP range | Population |
|---|---|---|---|---|
| Guilin | 18–36 (median 24) | 85 M / 63 F | SBP IQR 105–124, 5 subjects ≥140 | Healthy young adults |
| Foggia | 18–86 | 63 M / 64 F | SBP 122 ± 16 | Normo- and hypertensive, light skin tones only |
| PPG-BP | 20–89 (mean 57) | ≈1:1.08 | SBP 128 (80–182) | Hospital patients incl. hypertension and diabetes |
| ICU sets | Mostly older adults | Mixed | Wide, includes hypotension | Critically ill |

The pitch deck targets serving military personnel. Guilin's age profile is the closest
match, but its narrow BP range means it **under-represents hypertension** (the condition a
screening device most needs to catch). That is why the older, wider-range Foggia and PPG-BP
cohorts are used as external tests.

## 7. Sensor placement

Every shortlisted BP-labelled dataset is **fingertip** except Auricle (earlobe, transmissive)
and BUT PPG (smartphone held to the ear, camera-based). **None is behind-the-ear reflective
PPG over the posterior auricular artery.** Fingertip PPG is not equivalent to behind-ear PPG:
vascular bed, perfusion, contact pressure, wave-reflection timing and motion coupling all
differ (see `DOMAIN_ADAPTATION.md`).

## 8. Licensing and accessibility

| Dataset | Downloadable here without credentials? | Commercial use | Attribution |
|---|---|---|---|
| Guilin | Yes | Yes (CC BY 4.0) | Required |
| Foggia | Yes | Yes (CC BY 4.0) | Required |
| PPG-BP | Yes | Yes (CC0) | Requested (cite Sci Data 2018) |
| Pulse Transit Time PPG | Yes | Yes, share-alike on derived DBs (ODbL) | Required |
| Auricle | Yes | Yes (ODC-By) | Required |
| CardioWave-Portable | **No**: PhysioNet credentialing + DUA | Research only | — |
| PulseDB | Yes (Box/Kaggle) | VitalDB-derived part **non-commercial** | Required |
| MIMIC-III/IV waveform | Yes | ODbL | Required |
| MIMIC-III-Ext-PPG | **No**: credentialed | Research | — |
| Aurora-BP | **No**: DUA application | Research | — |

No access restriction was bypassed. Credentialed datasets were documented, not downloaded.

## 9. Weighted scoring

Weights follow the project brief: PPG compatibility 20 %, BP label quality 20 %,
acquisition compatibility 20 %, size/diversity 15 %, sampling/quality 10 %,
access/licence 10 %, documentation 5 %. Each criterion is scored 0–5.

Rules applied so that size cannot outweigh incompatibility:
- *PPG compatibility* is 5 only for separately available red **and** IR channels. A single
  unspecified-wavelength pleth scores at most 2.
- *Label quality* favours cuff references over invasive ABP (the device is meant to agree
  with a cuff), and simultaneous over temporally adjacent readings.
- *Acquisition compatibility*: ear site 3, reflective fingertip 2, transmissive/ICU fingertip 1,
  smartphone camera 1.
- *Size* is capped at 5 for ≥ 500 subjects.
- **Gate:** any dataset scoring ≤ 1 on acquisition compatibility **and** ≤ 2 on PPG
  compatibility has its weighted total multiplied by 0.75.

| Dataset | PPG (20) | Label (20) | Acq. (20) | Size (15) | fs/quality (10) | Access (10) | Docs (5) | Weighted /5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Guilin dual-wave | 5 | 3 | 2 | 3 | 4 | 5 | 3 | **3.50** |
| Auricle | 3 | 4 | 3 | 2 | 5 | 4 | 5 | **3.45** |
| Foggia multiwave | 4 | 4 | 2 | 3 | 2 | 5 | 4 | **3.35** |
| Pulse Transit Time PPG | 5 | 3 | 2 | 1 | 5 | 4 | 4 | **3.25** |
| PPG-BP | 2 | 3 | 1 | 3 | 3 | 5 | 5 | 2.70 × 0.75 = **2.03** |
| CardioWave-Portable | 2 | 3 | 1 | 5 | 4 | 1 | 4 | 2.65 × 0.75 = **1.99** |
| PulseDB v2 | 2 | 2 | 1 | 5 | 3 | 3 | 5 | 2.60 × 0.75 = **1.95** |
| MIMIC-III matched | 2 | 2 | 1 | 5 | 3 | 4 | 3 | 2.60 × 0.75 = **1.95** |
| VitalDB | 2 | 2 | 1 | 5 | 4 | 3 | 4 | 2.65 × 0.75 = **1.99** |
| BUT PPG | 1 | 2 | 1 | 1 | 1 | 5 | 4 | 1.75 × 0.75 = **1.31** |
| UCI Cuff-less | 2 | 2 | 1 | 4 | 3 | 5 | 2 | 2.50 × 0.75 = **1.88** (and unusable: no subject IDs) |

(Weighted = Σ score × weight / 100. PPG-BP is gated too: one unspecified-site IR channel scores 2 for PPG and 1 for acquisition. It is still used, as an external IR-only test, because it is small, CC0 and older.)

## 10. Compatibility with MAX30102

The MAX30102 provides red (660 nm) and IR (880 nm) reflective PPG, 18-bit, at 50–3200 sps.
Guilin (raw red+IR ADC counts with DC) and Foggia (MAX86150 red+IR, reflective) are the
closest in signal type. PTT-PPG uses the sister chip MAX30101 (same 660/880 nm LEDs), but it
has only 22 subjects and activity-protocol BP. No dataset used a MAX30102 at the ear.

## 11. Compatibility with behind-ear placement

**None of the datasets is behind-ear.** Auricle's earlobe channel is the nearest proxy and is
recommended as the next download for a finger→ear transfer study. In-ear PPG literature
(Davies et al. 2020; Ferlini/Montanari EarSet 2023; in-ear BP, Physiol Meas 2021) shows
ear-site pulses have about 2–3× smaller AC amplitude than the finger and a broader systolic
peak, so models trained on fingertip morphology must be expected to shift.

## 12. Final recommendation and rankings

**Ranking A: best for initial ML experimentation (downloaded and used):**
1. Guilin dual-wave (training and subject-wise CV; red vs IR vs red+IR ablation)
2. Foggia multiwave (external test, red+IR, older and wider-BP cohort)
3. PPG-BP (external test, IR only, older hospital cohort)
4. Pulse Transit Time PPG (MAX30101 sister chip; future work)

**Ranking B: best for eventual RakshPulse hardware compatibility:**
1. *RakshPulse's own behind-ear MAX30102 recordings with simultaneous validated cuff*
   (does not exist yet; this is the required next step)
2. Auricle (earlobe PPG + CNAP + cuff, the only ear-site BP dataset found). **Downloaded and analysed**:
   finger-trained models do not transfer to the earlobe (MODEL_EVALUATION.md §F)
3. Guilin dual-wave (same LED wavelengths class and raw-count format as MAX30102)
4. Pulse Transit Time PPG (MAX30101 chip)

The two rankings differ on purpose. A dataset can be good for building a pipeline without
being evidence that the pipeline works at the ear.

## 13. Research limitations

- Auricle was downloaded in a follow-up session (healthy cohort first; see DATASET_ACQUISITION.md) and
  analysed in MODEL_EVALUATION.md §F. Pulse Transit Time PPG (2.9 GB) was not downloaded.
- The Guilin sensor site and LED wavelengths are **not stated by the dataset authors**. The
  "finger-contact, 660/940 nm" description comes from a third-party index and is unverified.
- Dataset facts for CP-PPG, Four-wavelength PPG and Aurora-BP come from landing pages and summaries and were not
  re-checked file by file. The VitalDB licence was re-checked in 3.0 (CC BY 4.0, PhysioNet v1.0.0).
- RakshPulse 3.0 additional-dataset review: VitalDB was adopted (open, CC BY 4.0, PPG + ECG + arterial line + real
  NIBP). BIDMC has no BP labels. MIMIC-IV Waveform v0.1 (ODbL) has only about 200 records and no curated cuff
  pairing. CardioWave-Portable and MIMIC-III-Ext are credentialed, so they were not used (access restrictions are
  not bypassed).
- Search coverage is limited to English-language sources and to what the search tools indexed.

## 14. References and official links

- Guilin dual-wave: https://data.mendeley.com/datasets/pz2zzr8vhm/1 (doi:10.17632/pz2zzr8vhm.1)
- Foggia: https://data.mendeley.com/datasets/gs7jvwtcgt/1; descriptor: https://pmc.ncbi.nlm.nih.gov/articles/PMC12682042
- PPG-BP: https://doi.org/10.6084/m9.figshare.5459299.v5; Liang et al., Sci Data 5:180020 (2018)
- Pulse Transit Time PPG: https://physionet.org/content/pulse-transit-time-ppg/1.1.0/
- Auricle: https://physionet.org/content/auricle/1.0.0/ (doi:10.13026/swme-vp69)
- VitalDB: https://physionet.org/content/vitaldb/1.0.0/ (doi:10.13026/czw8-9p62); Lee et al., Sci Data 9:279 (2022)
- CardioWave-Portable: https://physionet.org/content/cardiowave-portable/1.0.0/
- PulseDB: https://github.com/pulselabteam/PulseDB; Wang et al., Front Digit Health 2023
- MIMIC-III Waveform Matched: https://physionet.org/content/mimic3wdb-matched/1.0/
- MIMIC-IV Waveform: https://physionet.org/content/mimic4wdb/0.1.0/
- UCI Cuff-Less BP: https://archive.ics.uci.edu/dataset/340
- BUT PPG: https://physionet.org/content/butppg/2.0.0/
- EarSet: https://zenodo.org/records/8142332
- MAX30102 datasheet: https://www.analog.com/media/en/technical-documentation/data-sheets/max30102.pdf
- Charlton P. "Photoplethysmography Datasets" list: https://peterhcharlton.github.io/post/ppg_datasets
- González et al. "Intensive care PPG datasets and ML for BP estimation: generalization not guaranteed", Front Physiol 2023
