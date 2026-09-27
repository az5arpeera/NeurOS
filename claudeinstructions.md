Technical Specification & Code Generation Prompt: Zero-Calibration Multi-Modal BCI Core
1. System Prompt Paradigm
Act as an expert Neurotech Software Architect and Elite Core Engineer specializing in biosignal processing, differential geometry, and deep learning architectures. Your task is to implement a production-ready, fully decoupled, universal software layer for Brain-Computer Interfaces (BCIs) that achieves absolute zero-calibration for new users using Riemannian Euclidean Alignment (EA) paired with a multi-modal convolutional neural network.
Strictly adhere to the architectural guidelines, mathematical constraints, and data engineering patterns specified below. All generated code must be clean, heavily commented, production-grade, and include thorough error handling and data boundary checks.

2. Architecture & File Structure Topology
The project repository must be structured as a modular, highly decoupled Python package to facilitate seamless execution, testing, and third-party app integration:

universal_bci_layer/
│
├── README.md                           # System overview and local deployment guide
├── requirements.txt                    # Pinning dependencies (numpy, scipy, mne, moabb, torch, websockets)
│
├── core/
│   ├── __init__.py
│   ├── ingestion.py                    # Real-time LSL fabric ingestion bridge and data frame workers
│   ├── geometry.py                     # Riemannian manifold utility math and Euclidean Alignment filters
│   └── models.py                       # PyTorch implementation of Multi-Modal Extended EEGNet (HEGNet/PhysioOmni hybrid)
│
├── pipelines/
│   ├── __init__.py
│   ├── train_offline.py                # Cross-subject training loop, data caching, and serialization
│   └── live_inference.py               # Asynchronous WebSocket engine orchestrating live translation nodes
│
└── tests/
    ├── __init__.py
    └── test_pipeline.py                # Validation test suite utilizing automated mock streaming arrays


3. Core Module Engineering Specifications
A. Ingestion Core (core/ingestion.py)
Protocol Requirement: Implement a robust client interface using pylsl or BrainFlow to handle high-velocity multi-stream synchronization.
Multi-Modal Structuring: Define a deterministic channel mapping framework to handle varying configurations dynamically. The system must accommodate a baseline combination of 8 EEG channels, 2 EOG channels (for ocular tracking/artifact handling), and 2 EMG channels (for motor reinforcement and muscle artifact cross-referencing).
Data Window Buffering: Implement a high-performance rolling ring-buffer using thread-safe structures or optimized NumPy operations to yield sliding matrices of shape (Total Channels, Samples Per Window).
B. Geometric Alignment Filter (core/geometry.py) 

Manifold Math: Build an explicit processing block using scipy.linalg to translate raw observation coordinate sets onto the Symmetric Positive-Definite (SPD) Riemannian manifold.

Euclidean Alignment (EA) Execution:
Ingest an initialization block of uncalibrated biometric streams
(\[X_{i}\]) representing the first few seconds of a session.Compute the spatial covariance grid for each epoch: \(P_i = \frac{1}{T-1}X_i X_i^T\)

Calculate the reference mean state \[\={P}\] across the initialization block.Compute the spatial alignment tensor utilizing symmetric eigenvalue metrics: \(R = \bar{P}^{-1/2}\).Apply the projection matrix linearly to all subsequent real-time streaming arrays on the fly: \(\tilde{X}_i = R X_i\).

Validation Gate: Verify that the matrix trace of the resulting post-alignment covariance matrix converges smoothly to the absolute geometric origin (the Identity Matrix, \[I\]). 


C. Multi-Modal Deep Neural Network (core/models.py)
The Problem: Standard EEGNet is structurally locked to pure cortical configurations and maps channel spatial relationships using single-dimensional depthwise operations optimized solely for scalp coordinates.
The Architecture Fix: Implement a generalized, multi-modal extension of the architecture in PyTorch (modeled after hybrid architectures such as HEGNet).
Layer Topology:
Modality Split Input Block: Accept incoming aligned frames of shape (1, Channels, Samples). Segment the input array along the row index to route EEG, EMG, and EOG signals into parallel feature pathways.
Parallel Temporal Convolutions: Deploy separate 1D convolutional layers across each modality to extract distinct frequency components (e.g., 8–30Hz bandpass kernels for EEG motor imagery patterns; high-frequency >40Hz tracking loops for EMG muscle burst activation).
Decoupled Depthwise Spatial Filtering: Apply an independent spatial convolution layer to the EEG path to map inter-channel cortical dependencies across the scalp topology without corrupting features with peripheral muscle or ocular data.
Dynamic Feature Fusion Hub: Concatenate the resulting specialized spatio-temporal feature maps from all three modalities along the filter dimension. Append a channel attention block or sub-network layer to weight the modalities adaptively based on the user's instantaneous signal-to-noise ratio (SNR).
Classification Head: Output discrete cross-entropy classification logs representing clean user execution parameters (STATE_NEUTRAL, INTENT_LEFT, INTENT_RIGHT, INTENT_SELECT).



4. Pipeline & Data Caching Execution Plans
A. Offline Training Engine (pipelines/train_offline.py)
Dataset Integration: Hook directly into the Mother of All BCI Benchmarks (MOABB) ecosystem API to automate dataset downloading, validation, and standard caching routines.
Target Datasets:
moabb.datasets.Dreyer2023A / Dreyer2023B: Extract synchronized multi-modal data streams combining EEG, EOG, and EMG signals recorded across diverse groups of healthy individuals.
moabb.datasets.Cho2017 / moabb.datasets.Forenzo2023: Pull high-density, multi-subject motor imagery coordinate arrays to evaluate the system's spatial generalization capacity.


Decoupled Train/Test Partition Strategy:
Implement a rigorous Leave-One-Subject-Out (LOSO) cross-validation framework.


For each training loop, cache the data from \(N-1\) subjects locally. Apply the core.geometry Euclidean Alignment engine to each subject independently to center their baseline matrices on the Identity origin, and aggregate these normalized tensors into a massive unified training array.

Train the PyTorch multi-modal classification model on this joint distribution until loss metrics stabilize. Save the resulting optimized weights as a static file (models/pre_trained_weights.pth).
Evaluate performance metrics exclusively on the remaining holdout subject's data to ensure the system can decode uncalibrated biological variations accurately.


B. Asynchronous Live Inference Node (pipelines/live_inference.py)
Production Deployment Pattern:
Initialize the core.models neural network layer and inject the pre-trained weights file (pre_trained_weights.pth). Lock the network states into a permanent frozen evaluation profile (model.eval()).
Spin up an asynchronous WebSocket server loop via websockets


Upon a new user connection, ingest the first 3 seconds of raw streaming arrays to compute the dynamic alignment projection matrix (\(R = \bar{P}^{-1/2}\)) locally.

Transition immediately to the live translation loop: capture incoming data windows, apply the fixed geometric alignment filter, execute the forward tensor pass through the frozen network, package the highest-probability command into a standard JSON event string, and broadcast it to connected client applications with a target processing latency under 30 milliseconds.



5. Explicit Code Constraints & Validation Checklist
Zero Structural Gaps: Do not use code placeholders (...) or leave functions undefined. Every script block must be self-contained, compile correctly, and handle missing dependencies or disconnected hardware data streams gracefully.
Rigorous Math Integrity: Ensure all matrix transformations track coordinate structures explicitly. Implement numeric shielding (such as Tikhonov regularization constants) across all covariance inversions to protect against mathematical underflow or singular matrix runtime errors.
Strict Performance Target: Implement data processing structures via vectorized NumPy and PyTorch code blocks to maintain high execution speeds and prevent buffer overruns during high-frequency live inference loops.

Your Strategic Approach to Shift Coding to Claude
When you load this architecture specification file into a clean conversation session with Claude, initiate the engineering execution phase using the following sequence of clear, step-by-step commands:
Command 1 (Environment Setup): "Review Section 2 and Section 3 of the attached architecture document. Generate a comprehensive requirements.txt file and implement the complete, production-grade core/geometry.py file containing the core Riemannian Euclidean Alignment logic. Ensure all matrix operations include explicit numeric shielding to prevent runtime failures."
Command 2 (Neural Network Implementation): "Review the multi-modal neural network architecture detailed in Section 3-C. Generate the complete core/models.py file using PyTorch. Implement the parallel multi-modal split pathways, the depthwise spatial convolutions, the dynamic feature fusion block, and the final classification head exactly as specified."
Command 3 (Offline Training Loop): "Review Section 4-A. Generate the pipelines/train_offline.py script. Use the MOABB framework to fetch data from the Dreyer2023A dataset, handle the leave-one-subject-out train/test data splitting strategy, apply the geometric alignment filter, and implement the model serialization training loop."
Command 4 (Live Server Node): "Review Section 4-B. Generate the pipelines/live_inference.py asynchronous server loop. Ensure it safely handles the initial calibration buffering phase, uses the pre-trained weights, and streams clean JSON event packets down the network fabric under 30ms of processing latency."


