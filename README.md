# Bert-Coref-Resolution-Lee-
A replicate of Official github of End-to-end Neural Coreference Resolution  
(https://arxiv.org/pdf/1707.07045.pdf)  
Use  this for setting  up the requrements and preparing  glove vectors/Elmo(https://github.com/kentonl/e2e-coref)  

For setting up the prince cluster to  support running the scripts : follow : https://github.com/ppriyank/Prince-Set-UP

**bert_end_2_end.py & train-bert_end2end.py**

Replaced bert model to generate embedings at run time to replace glove vectors and  elmo vectors in original paper  

Since Bert works in sequences, and original code is written using sentences as chunks, the sequence is converted into run time  splits of sesntences.  For detailed explanation go  to line  #323.  
For easier explanation of tensorflow code go to  https://stackoverflow.com/questions/34970582/using-a-variable-for-num-splits-for-tf-split/56015552#56015552 (my own answer)

## Original Span Generation  
![Original Span Generation](https://github.com/ppriyank/Bert-Coref-Resolution-Lee-/blob/master/2.jpg)

## Co-reference resolution  

![Co-reference resolution](https://github.com/ppriyank/Bert-Coref-Resolution-Lee-/blob/master/1.jpg)


## Multi-Tasking Appraoch  

![Multi-Tasking Appraoch](https://github.com/ppriyank/Bert-Coref-Resolution-Lee-/blob/master/3.png)


