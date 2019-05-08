# Bert-Coref-Resolution-Lee-
A replicate of Official github of End-to-end Neural Coreference Resolution  
(https://arxiv.org/pdf/1707.07045.pdf)  
(https://github.com/kentonl/e2e-coref)  


#bert_end_2_end.py

Replaced bert model to generate embedings at run time to replace glove vectors and  elmo vectors in original paper  

Since Bert works in sequences, and original code is written using sentences as chunks, the sequence is converted into run time  splits of sesntences.  For detailed explanation go  to line  #323. 
For easier explanation of tensorflow code go to  https://stackoverflow.com/questions/34970582/using-a-variable-for-num-splits-for-tf-split/56015552#56015552 (my own answer)
