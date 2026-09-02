Wilderness labs is creating a voice operated manual of the Tactical combat casualty care (TCCC) proceedures for USAF Pararescue (P.J.). Since a vast majority of military medical triage errors come from improper following of proceedures under battlefield stress, we aim to provide a hands free way for these elite operators to revisit their guidelines/checklists in the field. Since these operators must operate behind enemy lines for long periods of time, it is critical that the system must 1. be entirely local to the device, 2. operate extremely power efficiently. 

This repository is aimed at creating the most power efficient Speech-to-text pipeline possible on modern hardware, without sacrificing quality. To that aim, we will be exploring two competing pathways: 

1. Creating our own model (./custom). 
2. Finetuning existing models to be more power efficient (./finetune).

This machine is running NixOS and has access to an nvidia 5090 gpu for training and finetuning of any models.
