# LING-230-Single-Model-Disfluency-Detector

A deep learning-based disfluency detector for LING 230. Currently we (roughly) plan our architecture to be:
- some type of encoder (most likely wav2vec, frozen during training)
- a CNN
- fully connected layers to provide an ourput for each disfluency type
