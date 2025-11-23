# Scene-Graph Guided Latent Editing (NSG + CLIP FiLM + Inversion)

이 레포는 SGDiff를 **“Local: Scene Graph / Global: Reference Image”** 편집 파이프라인으로 재구성했습니다.

- Local (무엇·어디): Scene Graph → **NSG Encoder** → local tokens → U-Net cross-attn K,V  
- Global (어떻게 보일지): Reference image → **CLIP image encoder** → **FiLM**(γ, β) → U-Net feature modulation  
- LDM은 **DDIM inversion 기반 editing**에 사용. VAE/UNet/CLIP은 freeze, 학습 파라미터는 NSG + FiLM(+LoRA)만 엽니다.

## 변경 요약
- `ldm/modules/nsg_encoder.py`: Transformer 기반 NSG encoder (scene graph → local tokens).
- `ldm/modules/global_film.py`: CLIP image encoder(동결) + FiLM generator.
- `ldm/modules/diffusionmodules/openaimodel.py`: FiLM 적용, LoRA(rank 선택) 지원, cross-attn에 NSG local tokens 사용.
- `config_{vg,coco}.yaml`: cond stage = NSG, global stage = CLIP FiLM, UNet freeze+LoRA 옵션.

## 환경 (Linux 기준)
```bash
git clone https://github.com/YangLing0818/SGDiff.git
cd SGDiff
conda env create -f sgdiff.yaml
conda activate sgdiff
mkdir -p pretrained
# CLIP (openai) 라이브러리가 없으면 설치
pip install git+https://github.com/openai/CLIP.git
```

필수 패키지: `pytorch==1.12.1`(CUDA 11.3), `pytorch-lightning==1.4.2`, `clip`(openai), `einops`, `omegaconf` 등. `sgdiff.yaml`에 포함되어 있습니다.

## 필요한 파일 (수동 다운로드)
- **VQ-VAE (first stage)**: https://ommer-lab.com/files/latent-diffusion/vq-f8.zip  
  압축 풀어서 `pretrained/vq-f8-model.ckpt` 위치.
- **데이터**: VG/COCO scene graph 전처리는 `DATA.md` 참고. 이미지/어노테이션 경로는 `config_vg.yaml`/`config_coco.yaml`에서 수정.
- **CLIP**: openai `ViT-B/32`가 자동 다운로드(인터넷 필요). 오프라인이면 사전 캐시된 CLIP 가중치를 `$HOME/.cache/clip`에 두면 됩니다.

## 모델 설정 (핵심)
- **동결(freeze)**: VAE, UNet 본체, CLIP.  
- **학습(train)**: NSG encoder, FiLM MLP, LoRA(rank=4, cross-attn 선형층).
- **Local cond**: NSG local tokens (`dim=512`, `max_tokens=64`) → UNet cross-attn K,V.
- **Global cond**: CLIP image embedding (`dim=512`) → FiLM(γ,β) per block 채널.
- **Inversion-friendly**: DDIM/forward diffusion으로 `z_T`를 만들고, 역확산 시 SG/FiLM 조건만 바꿔 편집.

## 학습 실행 예시
```bash
# VG
python trainer.py --base config_vg.yaml -t --gpus 0,

# COCO
python trainer.py --base config_coco.yaml -t --gpus 0,
```
주요 설정(`config_vg.yaml` 예):
- `cond_stage_config`: `ldm.modules.nsg_encoder.NSGEncoder` (num_objs/preds 맞춰 수정).
- `global_stage_config`: `ldm.modules.global_film.CLIPGlobalEncoder` (모델명, device).
- `unet_config.params.film_embedding_dim=512`, `lora_rank=4`, `freeze_unet=true`, `freeze_first_stage=true`.
- `base_learning_rate`는 NSG/FiLM/LoRA만 학습하도록 5e-5로 상향.

## 데이터 입출력 포맷
데이터로더(VG/COCO)는 `(image, objs, boxes, triples, obj_to_img, triple_to_img)`를 반환합니다.
- **Local**: `(objs, boxes, triples, obj_to_img, triple_to_img)` → NSG encoder → `c_local`.
- **Global**: `image` → CLIP image encoder → `h_global` → FiLM(γ,β) & `c_global`(1 token) for cross-attn.
- **Latents**: image → VAE encoder → `z0`. 학습 시 표준 DDPM loss(MSE on noise).

## 편집(Inversion) 워크플로우 개요
1) 원본 이미지 `x_ref` → VAE encoder → `z0`.  
2) DDIM forward(or inversion) → `z_T`.  
3) 변경된 scene graph → NSG → `c_local'`; reference image → CLIP → FiLM(γ,β).  
4) 역확산(`z_T -> z_hat0`)에서 조건만 교체 → VAE decoder → 편집 결과.

현 레포에는 학습 루프만 포함됩니다. 서버에서 테스트할 때는 DDIM inversion 샘플러를 추가로 작성하거나 기존 `testset_ddim_sampler.py`를 참고해 위 과정을 구현하세요.

## 정리
- Local: Scene Graph → NSG → cross-attn  
- Global: Reference image → CLIP → FiLM  
- Freeze: VAE/UNet/CLIP, Train: NSG + FiLM(+LoRA)  
- Config와 경로만 맞추면 바로 학습 가능합니다.
