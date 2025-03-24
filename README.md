
# HAT

## Contents
* [수정한 부분](#수정한-부분)  
* [실행](#실행)  
  * [Data Preparation](#data-preparation)  
  * [Trainin a SuperTransformer](#train-a-supertransformer)  
  * [Evolutionary Search](#evolutionary-search)
  * [Train a SubTransformer](#train-a-supertransformer)
  * [Testing](#testing)  



## 수정한 부분
[errors](https://github.com/ei-ai/HAT/tree/main/errors) 참고

## Dependencies
<details>
<summary> Ububtu </summary>
<div markdown=1>

* OS: Ubuntu 22.04 LTS (Windows WSL 사용)
* Python = 3.9.21
* requirements.txt
    ```sh
    cffi==1.17.1
    click==8.1.8
    colorama==0.4.6
    coloredlogs==15.0.1
    ConfigArgParse==1.7
    Cython==3.0.11
    -e git+https://github.com/mit-han-lab/hardware-aware-transformers.git@70e5a279d080670208249fdd98ed731fa9bcc466#egg=fairseq
    fastBPE==0.1.1
    filelock==3.16.1
    flatbuffers==24.12.23
    fsspec==2024.12.0
    humanfriendly==10.0
    Jinja2==3.1.5
    joblib==1.4.2
    lxml==5.3.0
    MarkupSafe==3.0.2
    mpmath==1.3.0
    networkx==3.2.1
    numpy==2.0.2
    nvidia-cublas-cu12==12.4.5.8
    nvidia-cuda-cupti-cu12==12.4.127
    nvidia-cuda-nvrtc-cu12==12.4.127
    nvidia-cuda-runtime-cu12==12.4.127
    nvidia-cudnn-cu12==9.1.0.70
    nvidia-cufft-cu12==11.2.1.3
    nvidia-curand-cu12==10.3.5.147
    nvidia-cusolver-cu12==11.6.1.9
    nvidia-cusparse-cu12==12.3.1.170
    nvidia-nccl-cu12==2.21.5
    nvidia-nvjitlink-cu12==12.4.127
    nvidia-nvtx-cu12==12.4.127
    onnx==1.17.0
    onnxruntime==1.19.2
    packaging==24.2
    portalocker==3.1.1
    protobuf==5.29.2
    pycparser==2.22
    regex==2024.11.6
    sacrebleu==2.5.0
    sacremoses==0.1.1
    sympy==1.13.1
    tabulate==0.9.0
    tensorboardX==2.6.2.2
    torch==2.5.1
    tqdm==4.67.1
    triton==3.1.0
    typing_extensions==4.12.2
    ujson==5.10.0
    ```

</div>
</details>


## 실행 
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

    ```sh
    nohup \
    python ./train.py --configs=configs/wmt19.en-de/subtransformer/wmt19ende_npu@200ms.yml \
    --sub-configs=configs/wmt19.en-de/subtransformer/common.yml \
    > test_results/original/subtransformer/wmt19.en-de.txt 2>&1 \
    &
    ```
    </details>
    ```sh
    python train.py --configs=[subtransformer_architecture].yml --sub-configs=configs/[task_name]/subtransformer/common.yml
    ```
2. Convert a Model
    * `.pt` to `.onnx`
    ```sh
    python convert2onnx.py --configs=[subtransformer_architecture].yml --sub-configs==configs/[task_name]/convert_onnx/common.yml
    ```
    ```sh
    python convert2onnx.py --configs=configs/wmt14.en-de/convert_onnx/common.yml --sub-configs=configs/wmt14.en-de/convert_onnx/HAT_wmt14ende_xeon@204.2ms_bleu@27.6.yml
    ```
    * `.onnx` to `.rknn`
    ```sh
    python convert2rknn.py --onnx-name=[model_name]
    ```
    ```sh
    python convert2rknn.py --onnx-name=HAT_wmt14ende_xeon@204.2ms_bleu@27.6
    ```


### Testing
* NPU latency
    ```sh
    python train.py --rknn-name=[model_name] --latnpu -a=[arch]
    ```
    ```sh
    python train.py \
        --configs=configs/iwslt14.de-en/subtransformer/HAT_iwslt14deen_test.yml \
        --sub-configs=configs/iwslt14.de-en/subtransformer/common.yml \
        --rknn-model=iwslt14_de_en --latnpu 
    ```
