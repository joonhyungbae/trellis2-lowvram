# PLAN: trellis2-lowvram 조사 결과와 설계안 (M0)

작성 2026-10-28. 이 문서가 승인되기 전에는 구현하지 않는다.
표시 규칙: **[확인]** 은 코드·원문을 직접 읽고 확인한 것이다. **[추정]** 은 아직 측정하지 않은 값이다. **[미검증]** 은 사용자 GPU 실측이 필요한 것이다.

---

## 0. 사용자 결정 (2026-10-28)

| 질문 | 결정 |
|---|---|
| Q2 DINOv3 게이트 | 학생이 각자 Hugging Face 에서 승인받는다. 가중치를 나눠 주지 않는다 |
| Q3 wheel 출처 | B1(이 저장소 Actions 빌드)을 배포용으로 쓴다. 개발 중에만 B2 를 허용한다 |
| Q6 테스트 GPU | 별도 16GB 카드 없음. 개발 기계 RTX 4090 24GB 에서 **16GB 상한 모의**로 측정한다 |
| Q1 nvdiffrast 비상업 조건 | 대비하지 않는다. 수업·전시용 범위로 보고, README 에 결과물 상업 사용 제한만 알린다 |
| Q7 라이선스 | MIT |
| M1 Windows 설치 시험 | 사용자가 학생 PC 에 전달해 시험한다 |

Q6 결정에 따라 M2 완료 기준과 문구를 바꾼다. 측정은 `torch.cuda.set_per_process_memory_fraction` 으로 PyTorch 할당 상한을 묶어서 한다. README·BENCHMARKS 에는 "실제 16GB 카드에서 동작" 이 아니라 "**4090 에서 PyTorch 할당 상한 모의(14.5GB)로 동작, 실제 16GB 카드 미검증**" 으로 쓴다. 이 모의에서는 CUDA 컨텍스트(수백 MB)와 디스플레이 점유분이 상한 밖에 있다. 그래서 상한은 16GB 가 아니라 **14.5GB** 로 잡아 여유를 둔다. 실제 16GB 카드 제보가 들어오면 BENCHMARKS 에 그대로 더한다.

---

## 1. 조사한 것

| 대상 | 확인한 상태 |
|---|---|
| `microsoft/TRELLIS.2` | `main` 75fbf01 (2026-06-05). 파이프라인·샘플러·설정·setup.sh·o-voxel 을 읽음 |
| `microsoft/TRELLIS.2-4B` (HF) | `pipeline.json`, 파일 목록·크기, 게이트 없음, MIT |
| `IgorAherne/TRELLIS.2-stableprojectorz` | d5d38f1 (2026-04-29). `install.py` 와 `whl/` 의 cp311 Windows wheel 6개 |
| `visualbruno/ComfyUI-Trellis2` | b0e84c0 (2026-09-25). Torch 2.7/2.8/2.10 별 wheel, tiled decoder 옵션, `blackwell_fix.py` |
| 확장 라이선스 원문 | nvdiffrast, nvdiffrec(JeffreyXiang 포크), CuMesh, FlexGEMM |
| `ilintar/trellis2-gguf` | 1차 범위 밖이라 읽지 않음. M4 뒤 조사 문서로 |

개발 기계 정보: RTX 4090 24GB, 드라이버 575, Linux. 16GB 카드는 없다. 그래서 `torch.cuda.set_per_process_memory_fraction` 으로 16GB 상한을 흉내 내 미리 걸러 볼 수 있다. 다만 이 결과는 실제 16GB 카드의 측정을 대신하지 못한다(드라이버·디스플레이 점유분이 다르다). "16GB 에서 동작" 문구는 사용자 실측 뒤에만 쓴다.

## 2. upstream 에서 확인한 사실

### 2.1 upstream 에 이미 low_vram 모드가 있다 [확인]
`Trellis2ImageTo3DPipeline` 은 `low_vram=True` 가 기본값이다. 이 모드에서는 각 단계 직전에 해당 모델을 `.to(cuda)` 로 올리고 끝나면 `.cpu()` 로 내린다(`trellis2/pipelines/trellis2_image_to_3d.py`). `pipeline.cuda()` 는 장치만 기록하고 모델은 옮기지 않는다.

따라서 이 프로젝트의 몫은 "오프로딩을 새로 만드는 것"이 아니다. 아래 다섯 가지가 이 프로젝트가 할 일이다.
1. upstream 오프로딩에 **측정**을 붙인다. 단계별 peak VRAM 과 시간을 잰다.
2. 오프로딩 바깥에 남는 **활성값 피크**를 줄인다. 디코더 타일링, 해상도·토큰 상한, 텍스처 굽기 크기가 대상이다.
3. **필요한 모델만 읽는다.** upstream 은 8개 모델을 모두 CPU RAM 에 올린다.
4. OOM 을 잡아 단계 이름과 함께 알리고 낮춘 설정으로 다시 시도한다.
5. 설치를 자동화한다.

### 2.2 단계 구성 [확인]
`run()` 의 실제 순서는 다음과 같다. `pipeline_type` 은 `'512' | '1024' | '1024_cascade' | '1536_cascade'` 이고 HF 기본값은 `1024_cascade` 다.

| # | 단계 | 모델 (HF 경로) | 가중치 파일 | 비고 |
|---|---|---|---|---|
| 0 | 배경 제거 | `briaai/RMBG-2.0` (BiRefNet 구조) | 약 0.9GB [추정] | 입력이 투명 배경 RGBA 면 건너뜀 |
| 1 | 이미지 조건 | `facebook/dinov3-vitl16-pretrain-lvd1689m` | 1.21GB | 512 와 1024 두 번 실행(512 경로는 한 번) |
| 2 | sparse structure flow | `ss_flow_img_dit_1_3B_64_bf16` | 2.58GB | 12 step, CFG 는 cond/neg 를 **차례로** 두 번 실행(배치로 묶지 않음) |
| 3 | sparse structure decoder | `TRELLIS-image-large/ss_dec_conv3d_16l8_fp16` | 0.15GB | 64³ 밀집 → 32³ 로 max-pool |
| 4 | shape SLat flow 512 | `slat_flow_img2shape_dit_1_3B_512_bf16` | 2.58GB | |
| 4b | (cascade) 업샘플 + shape flow 1024 | `shape_dec` 의 `upsample` + `…_1024_bf16` | 0.95 + 2.58GB | 토큰 수가 `max_num_tokens=49152` 를 넘으면 해상도를 128씩 내림 |
| 5 | texture SLat flow | `slat_flow_imgshape2tex_dit_1_3B_{512,1024}_bf16` | 2.58GB | |
| 6 | shape decoder | `shape_dec_next_dc_f16c32_fp16` | 0.95GB | 메쉬 + 서브구조. `fill_holes()` |
| 7 | texture decoder | `tex_dec_next_dc_f16c32_fp16` | 0.95GB | |
| 8 | 내보내기 | `o_voxel.postprocess.to_glb` | 없음 | CuMesh 리메시·단순화, UV 펼침, **nvdiffrast 로 텍스처 굽기** |

### 2.3 단계별 예상 VRAM [추정, 전부 미측정]
가중치는 bf16/fp16 이다. 단, `convert_to` 는 블록 몸통만 바꾸므로 임베딩·출력층은 fp32 로 남는다.

| 단계 | 가중치 | 활성값 (512 경로) | 판단 |
|---|---|---|---|
| DINOv3-L | 약 1.2GB (fp32 로 올라감) | 작음 | 여유 |
| flow 1.3B 하나 | 약 2.6~3GB | 토큰 수에 비례. flash/xformers 면 어텐션은 선형 | 512 는 여유가 있을 것. 1024 cascade 는 토큰이 49k 까지 가서 불확실 |
| shape/tex decoder | 약 1GB | **업샘플마다 복셀 수가 늘어 피크 후보 1순위**. upstream 의 `low_vram` 플래그는 디코더에 선언만 있고 쓰이지 않는다 [확인] | 타일링 필요 여부를 측정으로 정함 |
| to_glb | 0 | 원본 메쉬(수백만 면), 리메시, 4096² 굽기 | 피크 후보 2순위 |

결론적으로 가중치는 16GB 안에 넉넉히 들어간다. 문제가 될 수 있는 것은 디코더와 후처리의 활성값이다. 이 표의 숫자는 M2 의 `benchmark` 로 채운다.

### 2.4 어텐션·sparse conv 백엔드 [확인]
- 환경변수 `ATTN_BACKEND`(밀집)와 `SPARSE_ATTN_BACKEND`(희소)를 **import 시점에** 읽는다. 따라서 래퍼는 upstream 을 import 하기 전에 환경변수를 정해야 한다.
- 밀집 어텐션은 `xformers | flash_attn | flash_attn_3 | sdpa | naive` 를 지원한다.
- **희소(varlen) 어텐션은 `xformers | flash_attn | flash_attn_3` 만 지원하고 SDPA 경로가 없다.** 요구사항의 "마지막 대안 SDPA" 를 희소 쪽에도 쓰려면 패치가 필요하다. SDPA 로 varlen 을 흉내 내려면 시퀀스별로 나누거나 블록 대각 마스크를 써야 하고, 메모리가 크게 는다. 그래서 1차는 **xformers 를 필수 바닥선**으로 두기를 제안한다(아래 4절).
- sparse conv 기본값은 `flex_gemm` 이다(`SPARSE_CONV_BACKEND`). spconv·torchsparse 도 고를 수 있다.

### 2.5 모델 접근 [확인]
| 모델 | 게이트 | 라이선스 |
|---|---|---|
| `microsoft/TRELLIS.2-4B` | 없음 | MIT |
| `microsoft/TRELLIS-image-large` (ss decoder) | 없음 | MIT |
| `facebook/dinov3-vitl16-pretrain-lvd1689m` | **있음, 수동 승인(manual)** | DINOv3 License |
| `briaai/RMBG-2.0` (upstream 기본 배경 제거) | **있음, 자동 승인** | **CC BY-NC 4.0 계열(비상업)** [원문 재확인 필요] |
| `ZhengPeng7/BiRefNet` | 없음 | MIT |

- DINOv3 는 HF 계정, 모델 페이지 신청, Meta 의 승인, 토큰 입력까지 거쳐야 한다. 승인이 **수동**이라 몇 시간에서 며칠이 걸릴 수 있다. 설치 안내에서 이 단계를 가장 먼저 시키고, 기다리는 동안 나머지 설치를 진행하게 만든다.
- 승인 없이 쓸 공식 대안은 찾지 못했다. 커뮤니티 재업로드본은 출처와 라이선스 조건이 불분명하므로 기본값으로 쓰지 않는다(질문 Q2).
- 배경 제거는 upstream 의 `BiRefNet` 클래스가 `model_name` 만 바꾸면 되는 구조다. 그래서 **기본값을 `ZhengPeng7/BiRefNet`(MIT, 게이트 없음)으로 바꿀 것을 제안**한다. 게이트 하나와 비상업 조건 하나가 함께 사라진다. 품질 차이는 측정해 봐야 안다.

### 2.6 CUDA 확장과 라이선스 [확인]
| 확장 | 쓰이는 곳 | 라이선스 | 바이너리 재배포 |
|---|---|---|---|
| FlexGEMM | sparse conv, `grid_sample_3d` (필수) | MIT | 가능 |
| CuMesh | `representations/mesh/base.py` import (필수) | MIT | 가능 |
| O-Voxel | 디코더의 메쉬 변환, `to_glb` (필수). upstream 저장소 안의 하위 폴더 | MIT (upstream) | 가능 |
| nvdiffrast | `to_glb` 의 텍스처 굽기 (GLB 내보내기에 필수), 렌더러 | **NVIDIA Source Code License (1-Way Commercial)**. 3.3절: 연구·평가 목적의 비상업 사용만 허용 | 라이선스 전문을 동봉하면 재배포 가능(3.1) |
| nvdiffrec (renderutils) | PBR 미리보기 영상 렌더러에서만 씀 | NVIDIA Source Code License, 비상업(연구·평가) | 같음 |
| flash-attn | 선택 | BSD-3 | 가능. 단 Windows 공식 wheel 은 없음 |
| xformers | 희소 어텐션 바닥선 | BSD-3 | PyTorch 인덱스에 Windows cu128 공식 wheel 이 있음 |

- **nvdiffrec 은 쓰지 않는다.** 미리보기는 GLB 를 Gradio `Model3D`(브라우저 three.js)로 보여 주면 된다. 그러면 비상업 확장 하나와 빌드 하나가 빠진다.
- **nvdiffrast 는 GLB 텍스처 굽기에 지금 필수다.** 결과물 사용 조건에 영향이 있으므로 질문 Q1 에 올렸다.
- Blackwell(sm_120): ComfyUI-Trellis2 는 "CuMesh 리메시가 sm_120 에서 깨진다" 며 CPU 마칭큐브 우회를 둔다. 이것은 그쪽 주장이고 우리는 아직 확인하지 않았다 [미검증]. CuMesh 를 `sm_120` 을 넣어 빌드하면 해결되는 문제인지부터 본다.

## 3. 메모리 전략 구현안

전부 `src/trellis2_lowvram/` 에 두고 upstream 은 고치지 않는다. 런타임에 감싸거나 바꿔 끼운다.

1. **선택적 로딩** (`pipeline.py`): `Pipeline.from_pretrained` 를 쓰지 않는다. `pipeline.json` 을 읽어 `pipeline_type` 에 필요한 모델만 `trellis2.models.from_pretrained` 로 올리고, 다운로드도 그 파일만 받는다(`hf_hub_download` 개별 호출). 효과는 다음과 같다. [추정]
   - 512 경로는 가중치 약 9.8GB(flow 3개, 디코더 3개)만 받는다. 전체는 16.2GB 다.
   - CPU RAM 사용이 절반 가까이 준다. 32GB RAM 권장의 근거가 여기서 나온다.
2. **단계 계측** (`memory.py`): `StageMonitor` 컨텍스트로 단계마다 아래 순서를 밟는다. 단계 경계는 upstream 메서드(`get_cond`, `sample_sparse_structure`, `sample_shape_slat*`, `sample_tex_slat`, `decode_shape_slat`, `decode_tex_slat`, `to_glb`)를 래핑해서 잡는다. 메서드 이름은 고정 커밋 기준이고, 바뀌면 시작할 때 검사해서 알린다.
   - `reset_peak_memory_stats` 로 시작한다.
   - 단계를 실행한다.
   - `max_memory_allocated`, `max_memory_reserved`, 경과 시간, 장치 전체 사용량(`mem_get_info`)을 기록한다.
   - `empty_cache` 로 끝낸다.
3. **오프로딩 강도** 세 단계:
   - `normal`: upstream `low_vram` 그대로.
   - `aggressive`: 각 단계 뒤 `gc.collect()` + `empty_cache()`. 조건 텐서(`cond_1024` 등)는 다음에 쓸 때까지 CPU 에 둔다.
   - `extreme`: flow 모델을 블록 단위로 올렸다 내린다(sequential offload). 느려지므로 마지막 수단으로만 쓴다. 1차에서는 필요가 측정으로 확인될 때만 구현한다.
4. **정밀도**: 가중치 dtype 은 체크포인트가 정한 대로 둔다(flow bf16, VAE fp16). 단, RTX 30 이상은 bf16 을 모두 지원하므로 따로 분기하지 않는다. DINOv3 는 fp32 로 올라가는데, fp16 으로 내렸을 때 결과가 같은지 확인한 뒤에만 바꾼다.
5. **할당자**: 래퍼 진입점에서 torch import 전에 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 를 건다. 이 설정은 Windows 에서 무시된다는 보고가 있다 [미검증]. 그래서 doctor 가 실제로 적용됐는지 확인해 로그에 남긴다.
6. **어텐션 선택** (`attention.py`): import 전에 순서대로 시도한다. flash_attn import + GPU 가 sm80 이상이면 `flash_attn`, 아니면 xformers import 시도 후 `xformers`. 둘 다 없으면 밀집은 `sdpa` 로 두고, 희소는 "설치 손상" 으로 보고 안내 뒤 종료한다(Q4). 선택 결과는 로그와 리포트에 남긴다.
7. **해상도·토큰 상한**: 기본 `pipeline_type='512'`. `1024_cascade` 는 실험적 옵션이다. 시작 전에 `mem_get_info` 로 남은 VRAM 을 보고 경고한다. `max_num_tokens` 는 프리셋 값으로 내려 잡는다(upstream 이 이미 토큰 수에 따라 해상도를 낮추는 장치를 갖고 있어서 이를 이용한다). `1536_cascade` 는 막는다.
8. **디코더 타일링**: M2 측정에서 디코더가 한계를 넘을 때만 넣는다. 공간을 겹치는 블록으로 나눠 디코딩하고 이어 붙이는 방식이며, ComfyUI-Trellis2 의 구현(MIT)을 읽고 따온다. 이음매 품질은 사용자 눈으로 확인받는다.
9. **OOM 처리**: 단계 래퍼가 `torch.cuda.OutOfMemoryError` 를 잡는다. 순서는 다음과 같다.
   - 해당 단계 모델을 CPU 로 내리고 캐시를 비운다.
   - `StageOOM(stage, settings)` 를 올린다.
   - CLI 는 낮춘 설정 제안을 보여 주고 `y/N` 을 받는다. 제안 순서는 텍스처 4096→2048→1024, 그다음 해상도 1024→512, 그다음 오프로딩 강도 올리기다.
   - UI 는 같은 제안을 단추로 보여 준다.
   - 프로세스는 죽지 않는다.
10. **후처리**: `to_glb` 의 `texture_size` 와 `decimation_target` 을 프리셋으로 내린다. upstream 예제는 굽기 전에 `mesh.simplify(16777216)` 를 부른다(nvdiffrast 한계). 이 순서를 그대로 지킨다.

## 4. 설치 전략

### 4.1 기준 조합 (제안)
**Python 3.11 + PyTorch 2.8.0 + CUDA 12.8 wheel(cu128)**. 이 조합을 고른 이유는 셋이다.
- RTX 50(sm_120) 을 지원하는 첫 계열이다.
- 참고 저장소 두 곳이 같은 조합으로 Windows wheel 을 만들어 실제로 쓰고 있다.
- xformers 0.0.32.post2 cu128 공식 Windows wheel 이 있다.
드라이버는 cu128 런타임 기준 **R570 이상**이 필요하다. doctor 가 이것을 검사하고, 낮으면 NVIDIA 드라이버 페이지로 안내한다.

### 4.2 Windows: uv + 독립 Python (권장)
`install.bat` 은 다음 순서로 진행한다. 사용자는 아무것도 미리 설치하지 않아도 된다.
1. 저장소 폴더 안 `.tools/uv.exe` 로 uv 를 받는다(공식 독립 실행 파일, 관리자 권한 불필요).
2. `uv python install 3.11` 로 독립 Python 을 받는다(python-build-standalone).
3. `uv sync --frozen`: `uv.lock` 대로 torch cu128·xformers 등 순수 의존성을 설치한다.
4. CUDA 확장 4개(flex_gemm, cumesh, o_voxel, nvdiffrast)를 미리 빌드된 wheel 로 설치한다. 출처는 4.3 에서 정한다.
5. `doctor` 를 실행한다.
6. HF 토큰과 DINOv3 승인을 단계별로 안내한다.
7. 모델을 받는다. 이어받기가 되고 캐시 위치를 바꿀 수 있다.
로그는 `logs/install-<시각>.log` 에 남긴다.

- **위험**: uv 와 독립 Python 이 한글·공백이 들어간 사용자 경로(`C:\Users\홍길동\…`)에서 문제를 일으킬 수 있다 [미검증]. 설치 위치에 ASCII 가 아닌 글자가 있으면 doctor 가 경고하고 `C:\trellis2-lowvram` 같은 위치를 권한다. Windows 경로 길이 제한(260자)도 같이 검사한다.
- **대안 A: embedded Python zip.** stableprojectorz 가 이 방식이다. pip·venv 가 기본으로 빠져 있어 손이 더 가고, lock 파일과 맞추기도 어렵다.
- **대안 B: conda/Miniforge.** 무겁고(수 GB), uv 방침과 어긋난다.

### 4.3 CUDA 확장 wheel 의 출처 (Q3)
| 안 | 내용 | 장점 | 위험 |
|---|---|---|---|
| **B1 (권장)** | 이 저장소의 GitHub Actions(windows-2022 + CUDA 12.8 툴킷, GPU 불필요)에서 4개를 빌드한다. 고정 커밋에서 빌드해 GitHub Release 에 올린다. arch 목록은 `8.0;8.6;8.9;12.0+PTX` | 출처가 투명하고 재현할 수 있다. sm_120 을 직접 넣는다 | Actions 빌드 시간(CUDA 설치 포함 30~60분)과 처음 맞추는 수고. nvdiffrast wheel 은 라이선스 전문을 동봉해야 한다 |
| B2 | 커뮤니티 wheel(visualbruno·IgorAherne)을 원 주소에서 해시를 고정해 받는다 | 당장 빠르다 | 제3자 바이너리라 출처를 검증할 수 없다. 그쪽이 지우면 끊긴다. 우리 torch 조합과 정확히 맞는지 다시 확인해야 한다 |
| B3 | 사용자 PC 에서 빌드한다 | 재배포 문제가 없다 | VS Build Tools 와 CUDA Toolkit 이 필요해 "원클릭" 원칙을 깬다. 최후 수단 |

제안은 M1 에서 B1 을 만들고, B1 이 준비될 때까지만 개발용으로 B2 를 쓰는 것이다. 사용자 배포에는 B1 만 쓴다.

### 4.4 Linux
`install.sh` 도 같은 uv 흐름으로 간다. CUDA 확장은 B1 과 같은 방식으로 manylinux wheel 을 Actions 에서 만든다. 만들기 전까지는 사용자 PC 빌드도 허용한다(Linux 는 nvcc 를 conda/uv 로 받기 쉽다). `sudo apt` 는 쓰지 않는다. upstream 의 `pillow-simd`·`libjpeg-dev` 는 속도 최적화일 뿐이므로 빼고 일반 Pillow 를 쓴다.

### 4.5 업데이트
`update.bat/sh` 는 `git pull`(git 이 없으면 Release zip)을 받고 `uv sync --frozen` 을 한 뒤, 바뀐 확장 wheel 만 다시 깐다. upstream 커밋을 올리는 일은 사용자 업데이트가 아니다. 개발자가 `docs/UPSTREAM.md` 절차대로 한다.

### 4.6 upstream 고정 방법
upstream 은 `pyproject.toml` 이 없는 저장소다(`trellis2/` 를 경로로 import 한다). 그래서 다음처럼 둔다.
- `third_party/TRELLIS.2` 를 **git submodule** 로 둔다(커밋 75fbf01 고정, o-voxel 의 eigen 하위 submodule 포함).
- 래퍼가 시작할 때 `sys.path` 에 넣는다.
- git 이 없는 Windows 사용자를 위해 install 은 submodule 대신 해당 커밋의 GitHub tarball 을 받아 같은 자리에 푼다. 이렇게 하면 git 설치를 요구하지 않는다.
- 패치가 필요하면 `patches/*.patch` 를 설치 때 적용하고, 실패하면 설치를 멈춘다. 1차 목표는 패치 0개, 몽키패치만이다.

## 5. 폴더 구조 

```
trellis2-lowvram/
├── third_party/TRELLIS.2/         # submodule @75fbf01 (git 없으면 tarball)
├── patches/                       # 처음엔 비어 있음
├── wheels/                        # 빌드 스크립트만, 바이너리는 Release 에
│   └── build_windows.ps1, build_linux.sh, versions.toml
├── .github/workflows/
│   ├── tests.yml                  # CPU 단위 테스트
│   └── build-wheels.yml           # B1
├── src/trellis2_lowvram/
│   ├── bootstrap.py               # torch/upstream import 전 환경변수·sys.path 준비 (추가)
│   ├── env_check.py
│   ├── hf_access.py               # 토큰·DINOv3 승인 확인·안내 (추가)
│   ├── models.py                  # 선택적 다운로드·로딩 (추가)
│   ├── memory.py
│   ├── attention.py
│   ├── pipeline.py
│   ├── presets.py, presets.toml   # 값은 측정 후 확정
│   ├── export.py                  # GLB(to_glb), OBJ(trimesh)
│   ├── cli.py
│   ├── i18n/ko.json, en.json
│   └── ui/app.py
├── scripts/gpu_smoke_test.py
├── tests/
└── docs/PLAN.md, TROUBLESHOOTING.md, BENCHMARKS.md, UPSTREAM.md
```

## 6. 마일스톤별 작업과 완료 기준

### M1 설치·환경 점검
- `pyproject.toml` + `uv.lock`(torch cu128 인덱스 고정), ruff, `tests.yml`
- `env_check.py`: GPU 이름·compute capability·VRAM·드라이버(`nvidia-smi` 와 NVML, torch 없이도 동작), RAM, 디스크(모델 캐시 위치 기준), 경로 검사(ASCII·길이)
  - 판정: 통과 / 경고 / 중단. 문구는 i18n 키로만 쓴다.
- `hf_access.py`: 토큰 저장, `HfApi.auth_check` 로 DINOv3 접근 확인, 단계별 안내 문구
- `build-wheels.yml` (B1) 과 Release 업로드
- `install.bat/.sh`, `run`, `update`, 설치 로그
- 단위 테스트: env_check 판정표(가짜 nvidia-smi 출력), i18n 키 일치, 설정 파싱
- **완료 기준**: 사용자가 깨끗한 Windows PC 에서 `install.bat` 을 실행하면 `doctor` 가 통과하거나 해결 방법을 알려 준다.

### M2 메모리 최적화 CLI
- `models.py` 선택적 로딩, `memory.py` StageMonitor, `attention.py`, `pipeline.py`, OOM 재시도
- `cli.py generate / doctor / benchmark`, `export.py`
- `scripts/gpu_smoke_test.py`: 샘플 이미지(upstream `assets/example_image/T.png`)로 512 경로를 돌린다. JSON 과 마크다운으로 결과를 낸다.
- `--vram-cap GB` 옵션으로 모의 상한을 건다. 결과는 "모의" 로 표시해 PLAN 2.3 표를 채운다.
- **완료 기준**: 4090 에서 할당 상한 14.5GB 모의로 512 + 텍스처 생성이 성공하고 단계별 VRAM 리포트가 나온다(사용자 확인). 실제 16GB 카드는 미검증으로 표시한다.

### M3 웹 UI
- Gradio(버전은 upstream 과 같은 6.0.1 로 고정할지 M3 에서 확인), 프리셋 3개, 고급 설정, 멀티 seed 갤러리, 단계 진행·VRAM 표시, `Model3D` 미리보기, GLB/OBJ 다운로드, 한/영 전환(OS 언어 기본), `127.0.0.1` 기본, `--share` 는 명시할 때만
- **완료 기준**: 프리셋·멀티 seed·미리보기·GLB/OBJ 내보내기가 동작하고 한/영 전환이 된다.

### M4 문서·릴리스
- README(영)·README.ko, TROUBLESHOOTING, BENCHMARKS(실측만), THIRD_PARTY_LICENSES(2.5·2.6 표 기준), UPSTREAM, v0.1.0 태그
- GGUF 조사 문서(`docs/research/gguf.md`)

## 7. 미해결 질문

- **Q1. nvdiffrast 비상업 조건.** GLB 텍스처 굽기에 nvdiffrast 가 필수다. 그 라이선스는 "연구·평가 목적, 금전적 이득 없는 사용" 만 허용한다. 학생 작품 전시처럼 판매가 없는 쓰임은 대체로 이 안에 든다고 봅니다. 다만 결과물을 판매하는 쓰임까지 고려해야 하나요? 고려해야 한다면 굽기를 CPU 래스터화(xatlas + numpy)로 바꾸는 별도 경로가 필요하고, 일정이 늘어납니다.
- **Q2. DINOv3 게이트.** 수동 승인이 필요하다. 학생 수만큼 각자 신청하게 할지(정석), 아니면 운영자가 받은 가중치를 공유할지 정해야 한다. 공유는 DINOv3 License 의 재배포 조항을 원문으로 확인한 뒤에만 가능하다. 1차는 정석을 제안한다.
- **Q3. wheel 출처.** 4.3 의 B1(자체 Actions 빌드)을 승인하시는지. B1 이 준비되기 전 개발용으로 B2 를 쓰는 것도 괜찮은지.
- **Q4. 희소 어텐션 SDPA.** upstream 희소 어텐션에는 SDPA 경로가 없다. xformers 를 필수 바닥선으로 두고 SDPA 는 밀집 쪽에만 쓰는 것으로 요구사항(5.2)을 해석해도 될지.
- **Q5. 배경 제거 기본값.** upstream 기본인 RMBG-2.0(게이트·비상업) 대신 BiRefNet(MIT)을 기본으로 하고, RMBG-2.0 은 선택지로 둘지.
- **Q6. 테스트 GPU.** 사용자가 가진 16GB 카드의 모델명(4060 Ti 16GB, 4070 Ti Super, 4080, 5070 Ti, 5080 등)과 Windows 버전. 그리고 12GB 카드로도 시험할 수 있는지.
- **Q7. 라이선스.** MIT 로 결정(0절).
- **Q8. 출력 형식.** GLB·OBJ 외에 STL(3D 프린트용)이 학생 쓰임에 필요한지.
