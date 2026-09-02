# Synthetic fundus smoke dataset card

- **Source:** generated locally by `retina_lanse.data.synthetic`; no human or clinical data.
- **Licence:** repository-authored synthetic generator; no third-party images.
- **Patients/identifiers:** fictional deterministic patient identifiers only.
- **Images:** 72 RGB PNGs from 36 fictional patients, two images per patient.
- **Camera/device:** synthetic.
- **Dimensions/field of view:** 96 x 96 pixels with a simulated circular retinal field.
- **Labels:** none. Periodic bright/dark marks are generator controls, not medical annotations.
- **Missingness:** not applicable.
- **Permitted use:** software and reproducibility smoke tests only.
- **Prohibited inference:** clinical performance, lesion detection, diagnosis, or biological interpretation.
- **Split policy:** deterministic patient-level hashing; both images from one fictional patient remain together.

