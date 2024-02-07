# Automated Feature Labeling

This is a work-in-progress code base for testing an automated feature labeling technique.

Assuming we have found a particular monosemantic feature of a language model e.g. through the use of a sparse autoencoder, 
we still have the task of providing a human-understandable label for the concept that the feature represents. This approach aims to 
provide a label in the following way:

- For a dataset of text we assume we have the activation of the feature on each token in the text. In the simplest case we can represent the activations as 0s or 1s depending on whether they are greater than some threshold value.
- We give a prompt to a separate language model (in our case Llama) of the form "Answer the following question with either 'yes' or 'no'. Using the following piece of text for context |text|, does the word |token| represent the concept |concept|? Answer: "
- The |concept| token, rather than being a one-hot vector, is initialised as a random vector of size d_vocab, softmax is applied to produce a probability distribution and this distribution gives us an embedding that is a linear combination of all of token embeddings.
- We then perform gradient descent on the |concept| vector according to a loss function comprised of three terms:
  - A cross entropy term comparing the final token prediction to the ground truth of whether the feature activates on the given token |token| from the reference text.
  - A term corresponding to the entropy of the |concept| vector so as to encourage learning a small number of important concepts rather than a large superposition.
  - A KL divergence term between the |concept| vector probability distribution the model's probability distribution for the token at that position. This is to help narrow the search space of plausible concepts.

 The idea is that gradient descent learns the concept that allows the model most accurately predict the feature activation on different tokens of the dataset.

 This project was started during the ARENA hackathon.
  
