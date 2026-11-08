# Authorized Cholec80 setup

Verified November 7, 2026 against https://camma.unistra.fr/datasets/ . CAMMA currently provides a download link and lists CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/). This does not grant unrestricted commercial use or waive privacy obligations. Review current terms at the source, obtain organizational authorization, and preserve required attribution. No patient footage was downloaded for this project. No data access agreement was submitted or accepted on the user's behalf.

The publisher archive responded to a metadata-only HTTP HEAD request with a size of 74,916,858,781 bytes (about 75 GB). Allow additional space for extraction and preprocessing; the archive alone would consume most of the free disk on the verification machine. Download manually only when authorized. Place files locally:

```
data/cholec80/videos/video01.mp4
...
data/cholec80/phase_annotations/video01-phase.txt
```

Annotation TSV columns are `Frame` and `Phase`; frame numbering starts at zero. Native phase keys are Preparation, CalotTriangleDissection, ClippingCutting, GallbladderDissection, GallbladderRetraction, CleaningCoagulation, GallbladderPackaging. The numeric ordering in the original build brief differed at indices 4 and 6; canonical labels are authoritative. Annotation files drive alignment; no phase progression is imposed. Custom parsers may supply a label mapping, but changing classifier labels also requires corresponding checkpoint/API changes.

Cholec80 is 25 FPS constant-rate. Sampling uses source frame index / FPS and latest annotation at or before the frame, with strict coverage checks. Variable frame-rate material needs a timestamp-aware decoder (PyAV integration is not included). Files with inconsistent frame counts fail closed. The initial extraction resizes to 224×224 and retains RGB pixels; ImageNet normalization is applied in the loader.

The conventional 01–40 development / 41–80 test partition is preserved, reserving 33–40 for validation and training on 01–32. This exact validation subset is a project choice, not a claim of a universally prescribed validation split. Cases remain intact. If patient identifiers indicate multiple cases for one patient, group and audit those cases before splitting; Cholec80 case IDs alone cannot establish patient independence. Check CAMMA's overlap guidance before combining datasets.

Extracted arrays and metadata remain local and must follow the same governance as the source footage. Remove both frames and embeddings when retention ends. Upload cleanup does not manage your offline training dataset.
