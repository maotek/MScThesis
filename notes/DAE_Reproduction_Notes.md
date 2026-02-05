## Info
These notes are about the obstacles I came across while trying to integrate DepthAnyEvent within my own evaluation pipeline.

- Different input width and height to the module during infer.
    -
    The original DAv2 only supported a singular input_size (default=518). However, the DAE authors tweaked it and fed in an input_size_width of 350 and input_size_height of 266
- inv_prediction and activation
    - 
    The authors tweaked DAv2 by incorporating different activation fucntions: softplus, sigmoid, and relu. Also they have added an inv_prediction parameter, which applies 1 / (prediction + 1e-6) to the output.
    
    - These inv_prediction and activation parameters are defined in two places: the config.json in the checkpoint/ folder, and the configs/dav2/dav2_dsec_test.json. They were differen within both files, but in the end, they used "relu" and "true".

- Different way of computing the mean overall score.
    - 
    The DepthAnyEvent authors computed first the mean per sequences, and then average those means. However, I did not know this and I computed the mean per frame: sum of all frames across all sequences divided by the total frames. This resulted in worse results due to the influence of longer sequences.

- Dataloaders effect?
    - 
    The original authors constructed a dataloader from the dataset per sequence. In my initial evaluation pipeline, I used the raw dataset (without dataloader), since I got problems with h5 pickling when using num_workers > 0. However, I fixed this.
    - Remark: There was also an issue with the tencode not being deterministic in the case of using the dataset (without h5 pickling). However, this issue is yet to be investigated.