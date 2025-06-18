
# HAT

## Contents
* [Introduction](#introduction)
* [Getting Started](#getting-started)
* [How to Run](#how-to-run)  
  * [Data Preparation](#data-preparation)  
  * [Trainin a SuperTransformer](#train-a-supertransformer)  
  * [Evolutionary Search](#evolutionary-search)
  * [Train a SubTransformer](#train-a-supertransformer)
  * [Testing](#testing)  
* [Contributors](#contributors)

## Introduction
  The size of Transformer-based models is ever increasing, making it infeasible to deploy a model on edge devices with severe resource constraints. In this paper, we propose a search technique that integrates NAS(Neural Architecture Search) and HPO(HyperParameter Optimization) to find a hardware-friendly Transformer architecture that satisfies both memory and latency constraints. Evaluation results on the Rockchip 3588 NPU board show that latency was reduced by up to 2.15 times under the strict memory constraint of the NPU. This study significantly enhances the practicality of Transformer-based applications on edge devices and offers guidance for model optimization on AI-specific hardware.
## Getting Started
```sh
git clone https://github.com/ei-ai/THANOS.git
cd THANOS
pip install -e .[dev]
```

## Dependencies
<details>
<summary> CPU </summary>

* OS: Ubuntu 22.04 LTS (aarch64)
* Python = 3.9.21
</details>

<details>
<summary> NPU Board </summary>

* OS: Debian (Radxa Rock 5B image)
* Python = 3.9.21
</details>


## How to Run
### Data Preparation
```sh
bash configs/[task_name]/get_preprocessed.sh
```
<details>
<summary>download</summary>

```sh
bash configs/wmt14.en-de/get_preprocessed.sh
bash configs/wmt14.en-fr/get_preprocessed.sh
bash configs/wmt19.en-de/get_preprocessed.sh
bash configs/iwslt14.de-en/get_preprocessed.sh
```
</details>


### Train a SuperTransformer
1. Train a model
    <details>
    <summary>train</summary>

        ```sh
        python train.py --configs=configs/wmt14.en-de/supertransformer/space0.yml
        python train.py --configs=configs/wmt14.en-fr/supertransformer/space0.yml
        python train.py --configs=configs/wmt19.en-de/supertransformer/space0.yml
        python train.py --configs=configs/iwslt14.de-en/supertransformer/space1.yml
        ```
    </details>

    ```sh
    python train.py --configs=configs/[dataset_name]/supertransformer/space0.yml
    ```

    <details>
    <summary>download</summary>

        ```sh
        python download_model.py --model-name=HAT_wmt14ende_super_space0
        python download_model.py --model-name=HAT_wmt14enfr_super_space0
        python download_model.py --model-name=HAT_wmt19ende_super_space0
        python download_model.py --model-name=HAT_iwslt14deen_super_space1
        ```
    </details>

    ```sh
    python download_model.py --model-name=[model_name]
    ```


2. Convert a model    
    <details>
    <summary>convert</summary>
    
    * `.pt` to `.onnx`
    ```sh
    python convert2onnx.py --configs=configs/wmt14.en-de/convert_onnx/super.yml
    python convert2onnx.py --configs=configs/wmt14.en-fr/convert_onnx/super.yml
    python convert2onnx.py --configs=configs/wmt19.en-de/convert_onnx/super.yml
    python convert2onnx.py --configs=configs/iwslt14.de-en/convert_onnx/super.yml
    python convert2onnx.py --configs=configs/wmt14.en-de/convert_onnx/super.yml   --enc
    python convert2onnx.py --configs=configs/wmt14.en-fr/convert_onnx/super.yml   --enc
    python convert2onnx.py --configs=configs/wmt19.en-de/convert_onnx/super.yml   --enc
    python convert2onnx.py --configs=configs/iwslt14.de-en/convert_onnx/super.yml   --enc
    python convert2onnx.py --configs=configs/wmt14.en-de/convert_onnx/super.yml   --dec
    python convert2onnx.py --configs=configs/wmt14.en-fr/convert_onnx/super.yml   --dec
    python convert2onnx.py --configs=configs/wmt19.en-de/convert_onnx/super.yml   --dec
    python convert2onnx.py --configs=configs/iwslt14.de-en/convert_onnx/super.yml   --dec
    ```

    * `.onnx` to `.rknn`
    ```sh
    python convert2rknn.py --onnx-name=wmt14_en_de
    python convert2rknn.py --onnx-name=wmt14_en_fr
    python convert2rknn.py --onnx-name=wmt19_en_de
    python convert2rknn.py --onnx-name=iwslt14_de_en
    python convert2rknn.py --onnx-name=wmt14_en_de   --enc
    python convert2rknn.py --onnx-name=wmt14_en_fr   --enc
    python convert2rknn.py --onnx-name=wmt19_en_de   --enc
    python convert2rknn.py --onnx-name=iwslt14_de_en   --enc
    python convert2rknn.py --onnx-name=wmt14_en_de   --dec
    python convert2rknn.py --onnx-name=wmt14_en_fr   --dec
    python convert2rknn.py --onnx-name=wmt19_en_de   --dec
    python convert2rknn.py --onnx-name=iwslt14_de_en   --dec
    ```
    </details>

    ```sh
    python convert2onnx.py --configs=configs/[task_name]/convert_onnx/[search_space].yml [--enc|--dec]
    python convert2rknn.py --onnx-name=[model_name] [--enc|--dec]
    ```

    * download
    ```
    gdown --folder https://drive.google.com/drive/folders/1dB2Wha-Sl2I_qBM_Ty01RXBfCRXOHt5L
    ```



### Evolutionary Search  
1.  Generate a latency dataset
    <details>
    <summary>generate</summary>

    ```sh
    python latency_dataset.py --latnpu --configs=configs/iwslt14.de-en/latency_dataset/npu.yml
    python latency_dataset.py --latnpu --configs=configs/wmt14.en-de/latency_dataset/npu.yml
    python latency_dataset.py --latnpu --configs=configs/wmt14.en-fr/latency_dataset/npu.yml
    python latency_dataset.py --latnpu --configs=configs/wmt19.en-de/latency_dataset/npu.yml
    ```
    </details>

    ```sh
    python latency_dataset.py --configs=configs/[task_name]/latency_dataset/[hardware_name].yml
    ```

    * download
    ```sh
    gdown --folder https://drive.google.com/drive/folders/1ejT4pdvw0VM6Y0XICrnQ-iWwYaOLjxRp
    ```
2. Train a latency predictor
    <details>
    <summary>npu example</summary>

    ```sh
    python latency_predictor.py --configs=configs/iwslt14.de-en/latency_predictor/npu.yml
    python latency_predictor.py --configs=configs/wmt14.en-de/latency_predictor/npu.yml
    python latency_predictor.py --configs=configs/wmt14.en-fr/latency_predictor/npu.yml
    python latency_predictor.py --configs=configs/wmt19.en-de/latency_predictor/npu.yml
    ```
    </details>

    ```sh
    python latency_predictor.py --configs=configs/[task_name]/latency_predictor/[hardware_name].yml
    ```
    
3. Run evolutionary search with a latency constraint  
    <details>
    <summary>npu example</summary>

    ```sh
    python evo_search.py --configs=configs/wmt14.en-de/supertransformer/space0.yml --evo-configs=configs/wmt14.en-de/evo_search/wmt14ende_npu.yml
    python evo_search.py --configs=configs/wmt14.en-fr/supertransformer/space0.yml --evo-configs=configs/wmt14.en-fr/evo_search/wmt14enfr_npu.yml
    python evo_search.py --configs=configs/wmt19.en-de/supertransformer/space0.yml --evo-configs=configs/wmt19.en-de/evo_search/wmt19ende_npu.yml
    python evo_search.py --configs=configs/iwslt14.de-en/supertransformer/space1.yml --evo-configs=configs/iwslt14.de-en/evo_search/iwslt14deen_npu.yml
    ```
    </details>

    ```sh
    python evo_search.py --configs=[supertransformer_config_file].yml --evo-configs=[evo_settings].yml
    ```



### Train a Searched SubTransformer
1. Train a Model
    <details>
    <summary>train</summary>
    
    * original
    ```sh
    python ./train.py --configs=configs/iwslt14.de-en/subtransformer/iwslt14deen_npu@200ms.yml \
    --sub-configs=configs/iwslt14.de-en/subtransformer/common.yml 
    python ./train.py --configs=configs/wmt14.en-de/subtransformer/wmt14ende_npu@200ms.yml \
    --sub-configs=configs/wmt14.en-de/subtransformer/common.yml 
    python ./train.py --configs=configs/wmt14.en-fr/subtransformer/wmt14enfr_npu@200ms.yml \
    --sub-configs=configs/wmt19.en-de/subtransformer/common.yml 
    python ./train.py --configs=configs/wmt19.en-de/subtransformer/wmt19ende_npu@200ms.yml \
    --sub-configs=configs/wmt19.en-de/subtransformer/common.yml 
    ```

    * test(qkv=64)
    ```sh
    python ./train.py --configs=configs/iwslt14.de-en/subtransformer/iwslt14deen_npu_test0@200ms.yml \
    --sub-configs=configs/iwslt14.de-en/subtransformer/common64.yml 
    python ./train.py --configs=configs/wmt14.en-de/subtransformer/wmt14ende_npu_test0@200ms.yml \
    --sub-configs=configs/wmt14.en-de/subtransformer/common64.yml 
    python ./train.py --configs=configs/wmt14.en-fr/subtransformer/wmt14enfr_npu_test0@200ms.yml \
    --sub-configs=configs/wmt19.en-de/subtransformer/common64.yml 
    python ./train.py --configs=configs/wmt19.en-de/subtransformer/wmt19ende_npu_test0@200ms.yml \
    --sub-configs=configs/wmt19.en-de/subtransformer/common64.yml 
    ```
    </details>

    ```sh
    python train.py --configs=[subtransformer_architecture].yml --sub-configs=configs/[task_name]/subtransformer/common[qkv_size].yml
    ```
2. Convert a Model
    * `.pt` to `.onnx`
    ```sh
    python convert2onnx.py --configs=[subtransformer_architecture].yml --sub-configs==configs/[task_name]/convert_onnx/common[qkv_size].yml
    ```

    * `.onnx` to `.rknn`
    ```sh
    python convert2rknn.py --onnx-name=[model_name]
    ```


### Testing
* NPU latency
    ```sh
    python train.py --rknn-name=[model_name] --latnpu -a=[arch]
    ```

## Contributors
|[Minseo Kim](https://github.com/440g)|[Suhyeon Kim](https://github.com/holyB2)|[Jiyeon Ha](https://github.com/lina4544)|
|:---:|:---:|:---:|
---