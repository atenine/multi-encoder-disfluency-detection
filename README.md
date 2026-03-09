# LING-230-Single-Model-Disfluency-Detector

A deep learning-based disfluency detector for LING 230. Currently we plan our architecture to roughly follow Mohapatra et a.l 2021:
- some type of encoder (most likely wav2vec, frozen during training)
- a CNN
- fully connected layers to provide an ourput for each disfluency type

We also want to attempt applying these ideas:
- hybrid sampling
- using a multi-task learner for some disfluencies and single-task for others
- early stopping

If we have time, it would be nice to look into:
- passing multiple embeddings to the rest of the model, i.e., passing embeddings from both wav2vec and voc2vec so as to provide more detailed embeddings for disfluent vocalizations.
- if we use both multitask and single task learners, experimenting with a smaller model for blocks, since we have much less data for it than for the other disfluency types. 
