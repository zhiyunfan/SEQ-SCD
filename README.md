# Sequence-level Speaker Change Detection (SEQ-SCD) with Continuous Integrate-and-fire
SEQ-SCD is a code for our paper "SEQUENCE-LEVEL SPEAKER CHANGE DETECTION WITH CONTINUOUS INTEGRATE-AND-FIRE". It can automatically segment the input feature sequence according to the speaker change and provide segment-level speaker embeddings.

# Installation
pip install -r requirements.txt

# Get Started
This tutorial assumes that you have already followed the data preparation tutorial to prepare AMI corpus. 

# Configuration
To ensure reproducibility, SEQ-SCD relies on a configuration file defining the experimental setup:

```yaml

# A sequence-level speaker change detection model is trained. 
# Here, training relies on 4s-long audio chunks,
# batches of 128 audio chunks, and saves model to
# disk every one (1) day worth of audio.
task:
   name: SpeakerChangeDetection
   params:
      duration: 4.0
      batch_size: 128
      per_epoch: 1
      collar: 0.0
      non_speech: True

# Data augmentation is applied during training.
# Here, it consists in additive noise from the
# MUSAN database, with random signal-to-noise
# ratio between 5 and 20 dB
data_augmentation:
   name: AddNoise
   params:
      snr_min: 5
      snr_max: 20
      collection: MUSAN.Collection.BackgroundNoise

# Since we are training an end-to-end model, the
# feature extraction step simply returns the raw
# waveform.
feature_extraction:
   name: LibrosaMFCC
   params:
      sample_rate: 16000

architecture:
   name: pyannote.audio.models.SEQSCD
   params:
      decoder:
         struction: 'fc'
         tdnn_context: 1
      sincnet:
         skip: True
         out_channels: [80, 60, 60, 60, 60,60]
         kernel_size: [251, 5, 5, 5, 5,5]
         stride: [1, 1, 1, 1, 1,1]
         max_pool: [3, 3, 3, 3, 2,2]
      rnn:
         unit: LSTM
         hidden_size: 256
         num_layers: 2
         bidirectional: True
         dropout: 0
      cif:
         cif_weight_threshold: 0.99
         max_history: 2
         weight_active: 'crelu'
         relu_threshold: 1.0
         using_scaling: True
         using_bias_constant_init: True
         using_kaiming_init: True
         nonlinear_act: 'relu'
         normalize_scalar: 12.0
      codebook:
         is_used: True
         code_len: 100
         code_size: 512
         attention_size: 512
         num_attention_heads: 8
         attention_dropout: 0.1
      loss_cfg:
         count_weight: 1.0
         spk_loss_weight: 50.0
         down_rate: 8
         plot_speaker: True
         spk_loss_type: 'focal_bce'
         num_spk_class: 136
         speaker_embedding_path: '/opt/tiger/fanzhiyun/code/SEQ-SCD-book/view/speaker_embedding.txt'
scheduler:
  name: TriStageScheduler
  params:
     learning_rate: 0.0001
     peak_lr: 0.0001
     warmup_steps: 100
     hold_steps: 13000
     decay_steps: 13000
```
# Acknowledge
The SEQ-SCD borrows a lot of codes from pyannote-audio, pyannote-metric, pyannote-core and pyannote-pipeline.
