# Anonymous Submission

## Datasets

We evaluate on the following benchmarks. Please download each dataset from the links below and place the files under `data/` as described.

**3DSRBench** ([Link](https://huggingface.co/datasets/ccvl/3DSRBench)):
We use the `orientation_on_the_left` and `orientation_in_front_of` types.
Place the files under `data/3dsrbench/`.

**OmniSpatial** ([Link](https://huggingface.co/datasets/qizekun/OmniSpatial)):
We use the `allocentric` subset under the Perspective Taking category.
Place the files under `data/omnispatial/`.

**SpatialMQA** ([Link](https://huggingface.co/datasets/liuziyan/SpatialMQA)):
We use the `test` split.
Place the files under `data/spatialmqa/`.

**ViewSpatial-Bench** ([Link](https://huggingface.co/datasets/lidingm/ViewSpatial-Bench)):
We use the `relative_direction` subset under the Person Perspective category.
Place the files under `data/viewspatial_bench/`.

After downloading, organize the files as follows:

```
data/
├── 3dsrbench/
├── omnispatial/
├── spatialmqa/
└── viewspatial_bench/
```

## Requirements

- Python 3.9.21
- PyTorch 2.8.0 (CUDA)


## Installation
```bash
git clone https://github.com/anonymous11175250-a11y/anonymized.git
cd anonymized
pip install -r requirements.txt
```

## Usage
# Evaluation
```bash
python pcd_qwen25vl.py --model_size 7B 
```