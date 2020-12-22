# Learning from Lexical Perturbations for Consistent Visual Question Answering

**[Learning from Lexical Perturbations for Consistent Visual Question Answering](https://arxiv.org/abs/2011.13406)**
<br>
Spencer Whitehead, Hui Wu, Yi Ren Fung, Heng Ji, Rogerio Feris, Kate Saenko

This repository contains the ***VQA Perturbed Pairings (VQA P2)*** benchmark as well as code for the ***Question-Relatedness Regularized
Reasoning (Q3R)*** framework.

If you find any of the resources (e.g., code, data,...) in this repository useful, please cite:
``` tex
@article{whitehead2020vqap2,
    Author={Whitehead, Spencer and Wu, Hui and Fung, Yi Ren and Ji, Heng and Feris, Rogerio and Saenko, Kate},
    title={Learning from Lexical Perturbations for Consistent Visual Question Answering},
    journal={arXiv preprint arXiv:2011.13406},
    year={2020}
}
```


## VQA Perturbed Pairings (VQA P2) Benchmark

The benchmark is provided in `data/vqap2.questions.json` and `data/vqap2.annotations.json`.

These files are in the same format as the original [VQA v2.0](https://visualqa.org/) (Goyal et al., 2017) data.
However, each example has an `original_id`, which is the `question_id` of the original question from VQA v2.0, and a `perturbation` field that indicates what perturbation has been applied.


## Experiments

### Software Requirements
- python==3.6
- pytorch==1.2.0
- h5py
- pyyaml
- tqdm



### Setup and Preprocessing

1. Download the VQA v2.0 data: https://visualqa.org/download.html
2. Download [GloVe pretrained word embeddings](http://nlp.stanford.edu/data/glove.840B.300d.zip) and use `core/preprocess/preprocess_word_embeddings.py` to process it into a word-to-vector dictionary:
    ```
    {
        "word1": numpy.ndarray,
        "word2": numpy.ndarray,
        ...
    }
    ```
3. Download the visual features (36 per image) from the [BUTD repo](https://github.com/peteanderson80/bottom-up-attention) of [Anderson et al., 2018](https://arxiv.org/abs/1707.07998). Unzip and preprocess the features:
    ```
    python core/preprocess/preprocess_features.py --input_tsv_folder /path/to/feature_dir/ --output_h5 /output/path/traineval_feature.h5
    ```
4. Pack and preprocess the training data, where additional perturbed training data should have the same format as VQA P2. Any method can be used for generating perturbed data, but lexically perturbed and back-translated questions are available upon request.
    ```
    python pack_data.py --og_ques_file /path/to/v2_OpenEnded_mscoco_train2014_questions.json --og_ann_file /path/to/v2_mscoco_train2014_annotations.json --pert_ques_file /path/to/yourperturbed_train.questions.json --pert_ann_file /path/to/yourperturbed_train.annotations.json --out_ques_file /path/to/train2014.combined_questions.json --out_ann_file /path/to/train2014.combined_annotations.json
    python core/preprocess/preprocess_questions.py --glove_pt /path/to/generated/glove/pickle/file --input_questions_json /path/to/train2014.combined_questions.json --input_annotations_json /path/to/train2014.combined_annotations.json --output_filename /output/path/train_questions.h5 --vocab_json /output/path/vocab.json --mode train
    ```
5. Pack and preprocess the evaluation data:
    ```
    python pack_data.py --og_ques_file path/to/v2_OpenEnded_mscoco_val2014_questions.json --og_ann_file /path/to/v2_mscoco_val2014_annotations.json --pert_ques_file /path/to/vqap2.questions.json --pert_ann_file /path/to/vqap2.annotations.json --out_ques_file /path/to/eval_combined.questions.json --out_ann_file /path/to/eval_combined.annotations.json
    python core/preprocess/preprocess_questions.py --input_questions_json /path/to/eval_combined.questions.json --input_annotations_json /path/to/eval_combined.annotations.json --output_filename /your/output/path/eval_questions.h5 --vocab_json /path/to/vocab.json --mode eval
    ```

After these steps, you should have one directory that contains:
- train_questions.h5
- train_questions.h5.glove.p
- train_questions.h5.ids.json
- eval_questions.h5
- eval_questions.h5.ids.json
- vocab.json

Note, the preprocessed visual features can be placed in the same or a different directory as the above.



### Training
To train a model, run the following command:
```
python core/train.py --input_dir /path/to/preprocessed/files --save_dir /path/for/checkpoints --feature_h5 /path/to/traineval_feature.h5 --config /path/to/config
```
Here, `--input_dir` is the directory containing the peroprocessed files, `--save_dir` is the directory where the model files will be saved, `--feature_h5` is the path to the preprocessed visual features, and `--config` is a YAML config file. Config files for XNM are in `configs/`.



### Testing
```
python core/validate.py --input_dir /path/to/preprocessed/files --feature_h5 /path/to/traineval_feature.h5 --ckpt /path/to/checkpoint/model.pt --output_file /path/to/scores.log 
```



## Acknowledgements
- Our implementations are based on [XNM](https://github.com/shijx12/XNM-Net) and [StackNMN](https://github.com/ronghanghu/snmn).
