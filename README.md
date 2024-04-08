## Word2Eq

#Task and Dataset
The intention of the task is to attempt to build an NLP solver for Math Word Problems (MWP). The proposed dataset will be the SVAMP Dataset, originally introduced in the paper, “Are NLP Models really able to Solve Simple Math Word Problems?” (Patel et. al, 2021). The SVAMP dataset is a challenge dataset, providing different variations of similar one-unknown math word problems (grades four and lower), as well as their corresponding equation.

#Approach
We intend to approach this task by viewing the MWP solver as a sequence-to-sequence translation problem. In essence, our model will translate the word problem, written in English natural language, to a set of operator and operand tokens, with the “=” sign acting as a stop token. To best achieve this, we’ll be utilizing the transformer architecture, due to its powerful self-attention mechanism and its ability to perform seq2seq encoding & decoding tasks.
