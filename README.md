# Document-Level Simplification

This repository contains the code for ACL2023 paper: [SWiPE: A Dataset for Document-Level Simplification of Wikipedia Pages]().

<p align="center">
  <img width="500" src="images/SWiPE_Example_Sample.png">
</p>

## Accessing the data

We release the full annotated dataset in the `data/` folder.

### Edit representation



### Data format

## Automatic Edit Identification

We release the BIC model, which is the best-performing model at the task of automatic edit identification, and the model that was used for silver annotating the portion of the dataset which was not annotated.

## Document-Level Simplifiers

We release two models we finetuned on the SWiPE dataset, which correspond to finetuned BART-large models, finetuned on the original SWiPE and the SWiPE-clean datasets. The models can be downloaded from the HuggingFace hub:

...



## Cite the work

If you make use of the code, models, or dataset, please cite our paper:
```
@inproceedings{laban2023swipe,
  title={SWiPE: A Dataset for Document-Level Simplification of Wikipedia Pages},
  author={Philippe Laban and Jesse Vig and Wojciech Kryscinski and Shafiq Joty and Caiming Xiong and Chien-Sheng Jason Wu},
  booktitle={Proceedings of the 61th Annual Meeting of the Association for Computational Linguistics},
  volume={1},
  year={2023}
}
```

## Contributing

If you'd like to contribute, or have questions or suggestions, you can contact us at plaban@salesforce.com.
All contributions are welcome!


