# Third-party materials

The root MIT license covers the CrowdSense team's original software and its
documentation. It does not replace upstream terms or grant rights in footage, model
weights, or the libraries listed below.

## Code

| Component | Upstream | License |
| --- | --- | --- |
| `backend/models/` (VGG-19 + transformer with learnable region attention) | [Boosting Crowd Counting via Multifaceted Attention](https://github.com/LoraLinH/Boosting-Crowd-Counting-via-Multifaceted-Attention), Lin et al., CVPR 2022 | MIT |
| VGG-19 ImageNet weights (downloaded at start-up) | [torchvision](https://github.com/pytorch/vision) | BSD-3-Clause |

The files in `backend/models/` are adapted from the MAN implementation. Please cite
the MAN paper if you use them.

Front-end libraries are loaded from public CDNs and are not vendored: Bootstrap
(MIT), Font Awesome Free (icons CC BY 4.0, code MIT), and Chart.js (MIT).

## Media

The videos in `media/` are demonstration clips and the CrowdSense renderings of them
(count, density-map, and overlay views). Source footage of
public places such as Masjid al-Haram and Times Square remains the property of its
original owners and is included for non-commercial research demonstration only. Do
not reuse it outside this project without checking the original terms.

## Model weights and data

The fine-tuned checkpoint and the Arab-context dataset (400 annotated images,
augmented to 2,000) are not distributed. Public benchmarks referenced in the report,
namely ShanghaiTech, UCF-QNRF, JHU-Crowd++, and NWPU-Crowd, must be obtained from their
owners under their own terms.
