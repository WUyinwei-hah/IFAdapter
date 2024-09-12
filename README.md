<div align="center">

<h1>IFAdapter: Instance Feature Control for Grounded Text-to-Image Generation</h1>

<div>
Yinwei Wu<sup>1,2</sup>&emsp;Xianpan Zhou<sup>1</sup>&emsp;Bing Ma<sup>1</sup>&emsp;Xuefeng Su<sup>1</sup>&emsp;Kai Ma<sup>1</sup>&emsp;Xinchao Wang<sup>2</sup><sup>&dagger;</sup>
</div>
<div>
    <sup>1</sup>Tencent PGC&emsp;
    <sup>2</sup>National University of Singapore&emsp;
    <sup>&dagger;</sup>corresponding author 
</div>


<!-- <div>
<a target="_blank" href="https://arxiv.org/abs/2403.20249">
  <img src="https://img.shields.io/badge/arXiv-2312.17142-b31b1b.svg" alt="arXiv Paper"/>
</a>

<a target="_blank" href="https://wuyinwei-hah.github.io/rrnet.github.io/">
  <img src="https://img.shields.io/badge/Project-Page-blue" alt="arXiv Paper"/>
</a>
</div>
</div> -->

---
## Overview
![overall_structure](./assets/fig1.png)

We introduce the **I**nstance **F**eature **Adapter**(IFAdapter) to to exert fine-grained control over the generation of multiple instances.

## Features
The IFAdapter is readily integrated with various community models and LoRAs!
![loras](./assets/lora.png)
We show the results of IFAdapter in combination with [PixelArt](https://civitai.com/models/120096/pixel-art-xl), [Lelo-Lego](https://civitai.com/models/92444/lelo-lego-lora-for-xl-and-sd15), [Claymation](https://huggingface.co/DoctorDiffusion/doctor-diffusion-s-claymation-style-lora), and [Bluepencil](https://civitai.com/models/119012/bluepencil-xl). We express our gratitude for these great work contributed by these communities!

## Quick Start
1.
```bash
git clone https://github.com/WUyinwei-hah/RRNet.git


cd RRNet

pip install -r requirements.txt

```

2.

Download the pretrained IFAdapter weight from, and place it under `./pretrained_models`.

## Generation
Make sure your
```python
python3 generate.py --data_folder ./Dataset/contain/ --save_folder ./generation_result  --device 0 
```
---

## Citation
```
@misc{wu2024relation,
      title={Relation Rectification in Diffusion Model}, 
      author={Yinwei Wu and Xingyi Yang and Xinchao Wang},
      year={2024},
      eprint={2403.20249},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```