# Sparse autoencoders for mechanistic interpretability of the DNA sequence-based model [Borzoi](https://www.nature.com/articles/s41588-024-02053-6) @ f(DNA) Calico

Large ML frameworks, such as the DNA sequence-based model Borzoi, ingest large amounts of data for training. 
Training data contain multitudes of features, and as the model is successful at prediction tasks and downstream benchmarks, 
it has extracted these features from the input sequence. 
We aim to use sparse autoencoders to decompose activations from the first few layers of the pre-trained model into 
monosemantic concepts that map to known and unknown transcriptional regulatory motifs.

We use the top K approach for sparsity described by [L. Gao et al.](https://cdn.openai.com/papers/sparse-autoencoders.pdf) to reconstruct 
the activations of the first few convoluational layers.
