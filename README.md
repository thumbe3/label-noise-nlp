# Towards Robustness to Label Noise in Text Classification via Noise Modeling

Code repository for the arxiv preprint [Towards Robustness to Label Noise in Text Classification via Noise Modeling](https://arxiv.org/abs/2101.11214)

The code for the Beta Mixture Model on the cross entropy loss of the model has been adapted from [https://github.com/PaulAlbert31/LabelNoiseCorrection](https://github.com/PaulAlbert31/LabelNoiseCorrection)

The data file should contain one sample per line in the format: 
* \<Sentence\> <\Tab> \<Label\>

The add_noise.py script changes this format to the following (one sample per line):
*  \<Sentence\> <\Tab> \<Noisy Label\> <\Tab> \<Did original label change\> <\Tab> \<Original Label\>
 
For example consider a sentence A with original label 2, if the noise is able to perturb this label to say 4 then the file will contain: “A <\Tab> 4 <\Tab> 1 <\Tab> 2” , else it will contain: “A <\Tab> 2 <\Tab> 0 <\Tab> 2".

The embedding file used in the code is the Wikipedia 2014 + Gigaword 5 (6B tokens, 400K vocab, uncased, 300d vectors, 822 MB download) file provided at https://github.com/stanfordnlp/GloVe



Consider citing our work if you find it useful:

```
@misc{garg2021robustness,
    title={Towards Robustness to Label Noise in Text Classification via Noise Modeling},
    author={Siddhant Garg and Goutham Ramakrishnan and Varun Thumbe},
    year={2021},
    eprint={2101.11214},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
