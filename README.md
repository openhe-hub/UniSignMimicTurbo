# UniSignMimic Project
## Intro
```mermaid
graph TD
    A[Input Video & Ref Picture] --> B(Pose Extraction - DWPose);
    B --> C(Retarget - Basic Transform);
    C --> D(TNet Rigid Transform for Body Pose);
    C --> E(NIF2D Non-rigid Transform for Face Pose);
    D --> F[Combine Body & Face Pose];
    E --> F;
    F --> G[Filter Low Confidence Keypoints & Add empty frames ];
    G --> H(MST - Masked-Sign-Transformer);
    H --> I(Mimicmotion - PoseNet + VAE + Diffusion);
    I --> J[Output Video];

    %% Styling
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style J fill:#ccf,stroke:#333,stroke-width:2px
```
***
## Runs
***
## Modules
### Pose Extraction (DWPose)
### Retarget (Basic Transform)
### TNet Rigid Transform (for Body Pose)
### NIF2D Non-rigid Transform (for Face Pose)
### MST (Masked-Sign-Transformer)
### Mimicmotion (PoseNet + VAE + Diffusion)
