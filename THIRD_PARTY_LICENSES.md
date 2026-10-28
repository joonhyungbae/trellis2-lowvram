# Third-party licenses / 제3자 라이선스

This repository's own code is MIT ([LICENSE](LICENSE)). It contains **no model weights and no
third-party binaries**: the installer downloads everything below from its official source onto
your PC. The table says what each part is for and the condition that matters most in practice.
Always check the linked license text; this page is a summary, not legal advice.

이 저장소의 코드는 MIT 입니다. 모델 가중치와 남의 바이너리는 들어 있지 않고, 설치 프로그램이 아래 것들을
공식 출처에서 내 PC 로 받습니다. 표는 각 구성 요소의 용도와 실제로 중요한 조건을 요약한 것입니다.
법률 자문이 아니므로 링크된 원문을 확인해 주세요.

## What it means for your models / 만든 모델을 쓸 때

- **nvdiffrast** bakes every GLB texture and allows **non-commercial (research or evaluation)
  use only**. Treat models made with this tool as for study, research and non-commercial
  exhibition. 텍스처를 굽는 **nvdiffrast 는 비상업(연구·평가) 용도만** 허용합니다. 만든 모델은 공부,
  연구, 비상업 전시에 써 주세요.
- **DINOv3** (image encoder) is under Meta's DINOv3 License, which you accept when you request
  access on Hugging Face. 이미지 인코더 **DINOv3** 는 Hugging Face 에서 신청할 때 동의하는 Meta 의
  DINOv3 License 를 따릅니다.
- Background removal uses **BiRefNet (MIT)**, not upstream's default RMBG-2.0 (non-commercial).
  배경 제거는 비상업 조건이 있는 RMBG-2.0 대신 **BiRefNet(MIT)** 을 씁니다.

## Models (downloaded at install / 설치할 때 받음)

| Model | Used for | License |
|---|---|---|
| [microsoft/TRELLIS.2-4B](https://huggingface.co/microsoft/TRELLIS.2-4B) | the 3D generation model | MIT |
| [microsoft/TRELLIS-image-large](https://huggingface.co/microsoft/TRELLIS-image-large) | sparse structure decoder | MIT |
| [facebook/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) | image encoder (gated, manual approval) | DINOv3 License |
| [ZhengPeng7/BiRefNet](https://huggingface.co/ZhengPeng7/BiRefNet) | background removal | MIT |

## Code and CUDA extensions / 코드와 CUDA 확장

| Component | Used for | License | Note |
|---|---|---|---|
| [microsoft/TRELLIS.2](https://github.com/microsoft/TRELLIS.2) | upstream pipeline (`third_party/`, unmodified) | MIT | |
| FlexGEMM | sparse convolution | MIT | prebuilt wheel from this repo's releases |
| CuMesh | mesh processing | MIT | prebuilt wheel |
| O-Voxel | mesh conversion, `to_glb` | MIT (TRELLIS.2 root LICENSE; the folder has none of its own) | prebuilt wheel |
| [nvdiffrast](https://github.com/NVlabs/nvdiffrast) | texture baking, previews | NVIDIA Source Code License (1-Way Commercial): **non-commercial research/evaluation only** | prebuilt wheel ships the license text (`wheels/build.py` enforces it) |

## Python packages (from PyPI / PyTorch index)

Versions are pinned in `uv.lock`. License values below were read from the installed packages'
metadata (Windows, 2026-10-29).

| Package | License |
|---|---|
| torch, torchvision | BSD-3-Clause |
| xformers | BSD |
| triton / triton-windows | MIT |
| transformers, huggingface-hub, safetensors, hf-xet | Apache-2.0 |
| gradio | Apache-2.0 |
| kornia, timm, opencv-python-headless | Apache-2.0 |
| numpy | BSD-3-Clause (and bundled permissive licenses) |
| pillow | MIT-CMU |
| pillow-heif | source BSD-3-Clause; **binary wheels GPLv2** (they bundle libheif and libde265, LGPLv3, and x265, GPLv2) |
| trimesh, einops | MIT |
| easydict | LGPL-3.0 |
| plyfile | GPL-3.0-or-later |
| tqdm | MPL-2.0 AND MIT |
| zstandard, psutil | BSD-3-Clause |
| nvidia-ml-py | BSD |
| [uv](https://github.com/astral-sh/uv) (installer tool, `.tools/`) | MIT or Apache-2.0 |

The GPL/LGPL packages (pillow-heif wheels, plyfile, easydict) are installed on your PC from PyPI by
the installer; this repository does not redistribute them.
GPL·LGPL 패키지(pillow-heif 휠, plyfile, easydict)는 설치 프로그램이 PyPI 에서 PC 로 받으며, 이 저장소가
재배포하지 않습니다.
