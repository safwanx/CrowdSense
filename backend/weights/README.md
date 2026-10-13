# Model weights

Checkpoints are not stored in this repository. Put the CrowdSense checkpoint here as
`model.pth`, or point the backend at another file:

```bash
export CROWDSENSE_WEIGHTS=/path/to/model.pth   # PowerShell: $env:CROWDSENSE_WEIGHTS="C:\path\to\model.pth"
```

The checkpoint must be a `state_dict` for `models.vgg_c.vgg19_trans`. The upstream
MAN checkpoints trained on JHU-Crowd++ and UCF-QNRF use the same architecture, so
they are a reasonable starting point if you do not have the fine-tuned CrowdSense
weights; download links are in the
[MAN repository](https://github.com/LoraLinH/Boosting-Crowd-Counting-via-Multifaceted-Attention#test).
