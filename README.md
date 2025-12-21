### End-to-end gene prediction with pure deep learning structure

The idea of this project is using deep learning approach to replace traditional Hidden Markov-model for gene prediction.    
In short human language, for example, the goal is replacing Augustus with cutting-edge deep learning model.     

There are certain pros for considering such action:     

1. Transformer have attention mechanism that reasonablily better than markov process.   
   (no longer solely depend on previous state)     
2. With certain deep learning mechanism, such as: pre-train, fine-tune, MoE.    
   The model no longer need to self-trained again for achieve higher accuary as normal HMM.   

Cons:
1. It's computational hard for running such program. Def need more memory space and time complexity.    
   Both require extremely nice engineering capability to handle them.     
2. It's extremely hard to develop and design such program. That's why no biologists did it so far.  


---


### Encoder (BERT)

The current stage of planning is not pre-trainning everything from stracth rather accepte a open-source model from hugging face.        
The choice is DNABert-v2, a pre-trained model that published weights on hugging face.   
That model is already trained for capture local pattern, motif through trainning.   
So, it would be the backbone of encoder block.  
For the first step of design such program is that we need a encoder-decoder transformer model that fine-tuned to *Ab Initio Prediction*.  

*Ab Initio Prediction* is the most basic functionality for such gene finder program. It's require able to generate prediction in form of GFF/GTF in terms of given DNA input sequences.   
Therefore, theoretically, for achieving this, there have to be a decoder part for generating (to transfer what have already learned) the ideal outputs.  


---


### Trainning nn

The last step is trainning the neural network using given dataset.  
All the nn is build up by stacking pytorch. So using pytroch package is necessary.  

This is how you can download all package for Apple Silicon.     

```zsh
conda create -n nameyouwannaput
conda install pytorch torchvision torchaudio -c pytorch -c conda-forge
```

Beside, you also need hugging face transformer package to load the pre-trained model for encoder.   

```zsh
conda install -c conda-forge transformers
```


---


### Unsupervised pre-trainning

Following the trainning convention of GPT, the first stage of trainning decoder is working on unsupervised pre-trainnning. To let the model capture basic understanding of words and semantic meaning of linguistics. In short, the ideal trainning would be input of DNA Sequence in tokenized format. After encoder transformer, those token would then feed into the decoder for later cross-attention to make autoregressive prediction to generate the whole GFF corpus. Ideally, this task should belongs to story completion in Natural Language Processing. The content of pre-trainning for such gene finder program is stayed inside Gong's mind as secret. But the public parameter for pre-trainnning is copied from public disclosed generative AI company.     

| Parameter             | Concise Data       | Detailed Description
| :-------------------: | :----------------: | :-------------------
| **Optimizer**         | Adam               | Uses the Adam optimizer with specific hyperparameters for large-scale stability.
| **Batch Size**        | 32k → 3.2M tokens  | Started at 32,000 tokens and gradually ramped up to 3.2 million tokens to stabilize early training.
| **Learning Rate**     | 0.6 × 10⁻⁴         | The peak learning rate specifically tuned for the 175B parameter model size.
| **LR Warmup**         | First 375M tokens  | A linear warmup period over the first 375 million tokens to prevent gradient divergence.
| **LR Decay**          | Cosine to 10%      | Uses a cosine decay schedule, reducing the rate to 10% of its peak over 260 billion tokens.
| **Weight Decay**      | 0.1                | Regularization term applied to the weights to prevent overfitting and improve generalization.
| **Gradient Clipping** | 1.0                | Global norm clipping at 1.0 to handle high-variance gradients and ensure training stability.

Since Gemini 3 pro and Claude don't have public disclosed for pre-trainning parameter and methodology. We just applied GPT-3 175B Model trainning parameter right away. Furthermore, there would be documentation on what kinda of species data we used to pre-train the model.  

For Prokaryotics:    

|  Species  |  Source   | Data
| :-------: | :-------: | :---
| E.Coli    |  NCBI     | GFF + FASTA
|

For Eukaryotics:  

|  Species  |  Chromosome 
| :-------: | :----------
