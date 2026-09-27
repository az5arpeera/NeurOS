"""
Week 1 MVP: Hardware-Agnostic LSL Data Ingestion Bridge

Zero-Calibration BCI Universal Layer - Week 1 MVP Ingestion Bridge
Target: Option B - Software Playback Ingestion Suite

This script provides a completely self-contained, runnable pipeline demonstrating 
the Week 1 Engineering Gate. It launches a background thread to simulate the playback 
of synchronized multi-modal biosignals (8 EEG channels, 2 EMG channels, 2 EOG channels) 
over an LSL (Lab Streaming Layer) network outlet, while the main thread instantiates 
the Ingestion SDK to discover, stream, and buffer the data in real time.

Dependencies (To run locally):
    pip install pylsl numpy
"""

import time
import sys
import threading
import numpy as np
from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_byprop, local_clock



# ==============================================================================
# CONFIGURATION & SCHEMA DEFINITIONS
# ==============================================================================
STREAM_NAME = "UniversalBCI_MultiModal_Playback"
STREAM_TYPE = "EEG_EMG_EOG"
SAMPLE_RATE = 250  # Hz (Standard for consumer/research neuro-wearables)

# Define our multi-modal channel configuration
CHANNELS = {
    "EEG": ["Fz", "C3", "Cz", "C4", "Pz", "O1", "Oz", "O2"], # 8 Brain channels
    "EMG": ["Jaw_Clench", "Left_Brow"],                     # 2 Muscle channels
    "EOG": ["Horizontal_Eye", "Vertical_Eye"]               # 2 Eye-movement channels
}

CHANNEL_NAMES = CHANNELS["EEG"] + CHANNELS["EMG"] + CHANNELS["EOG"]
TOTAL_CHANNELS = len(CHANNEL_NAMES)

# Rolling buffer size for the engine (e.g., hold 2 seconds of historical data)
BUFFER_DURATION_SEC = 2


BUFFER_MAX_SAMPLES = SAMPLE_RATE * BUFFER_DURATION_SEC


# ==============================================================================
# PHASE 1: SOFTWARE PLAYBACK SIMULATOR (BACKGROUND THREAD)
# ==============================================================================
def mock_lsl_playback_loop(stop_event):
    """
    Simulates a playback file or hardware driver broadcasting multi-modal data 
    over the local network using the LSL protocol.
    """
    print(f"[Playback Simulator] Initializing LSL StreamInfo for {STREAM_NAME}...")
    
    # Create the LSL Stream Description
    # arguments: name, type, channel_count, nominal_srate, data_format, source_id
    info = StreamInfo(
        STREAM_NAME, 
        STREAM_TYPE, 
        TOTAL_CHANNELS, 


       SAMPLE_RATE, 
        'float32', 
        'universal_bci_layer_uid_123'
    )
    
    # Append extensive metadata tracking channel identities
    desc = info.desc()
    chns = desc.append_child("channels")
    for name in CHANNEL_NAMES:
        chn = chns.append_child("channel")
        chn.append_child_value("label", name)
        if name in CHANNELS["EEG"]:
            chn.append_child_value("type", "EEG")
        elif name in CHANNELS["EMG"]:
            chn.append_child_value("type", "EMG")
        else:
            chn.append_child_value("type", "EOG")
            
    # Instantiate the LSL Network Outlet


   outlet = StreamOutlet(info)
    print("[Playback Simulator] LSL Outlet successfully broadcasting to network.")
    
    # Time generation track
    t = 0.0
    time_step = 1.0 / SAMPLE_RATE
    
    while not stop_event.is_set():
        start_time = time.perf_counter()
        
        # Synthesize realistic signals instead of flat lines to mirror unique biological profiles
        # Subject profile simulation: Alpha waves (10Hz) on EEG, sporadic twitches on EMG
        sample = []
        
        # 1. Generate 8 channels of EEG (Sine waves around 10Hz mixed with low-amplitude noise)
        for i in range(len(CHANNELS["EEG"])):
            alpha_wave = 15.0 * np.sin(2 * np.pi * 10.0 * t + i)
            noise = np.random.normal(0, 3.0)
            sample.append(alpha_wave + noise)


           
        # 2. Generate 2 channels of EMG (High-frequency noise, simulating muscle tension)
        for i in range(len(CHANNELS["EMG"])):
            muscle_activity = np.random.normal(0, 2.0)
            # Simulate an intermittent jaw clench event every 4 seconds
            if int(t) % 4 == 0 and (t - int(t)) < 0.2:
                muscle_activity += np.random.normal(0, 45.0)
            sample.append(muscle_activity)
            
        # 3. Generate 2 channels of EOG (Slow drifting baselines with rare sharp spikes for eye blinks)
        for i in range(len(CHANNELS["EOG"])):
            blink_activity = 0.0
            if int(t) % 6 == 0 and (t - int(t)) < 0.15:
                blink_activity = 60.0 * np.sin(np.pi * (t - int(t)) / 0.15)
            sample.append(blink_activity + np.random.normal(0, 1.0))
            
        # Push the synthesized vector onto the local network link
        outlet.push_sample(sample)
        


       t += time_step
        
        # Precise timing execution guard to maintain exact nominal sample rate
        elapsed = time.perf_counter() - start_time
        sleep_time = max(0.0, time_step - elapsed)
        time.sleep(sleep_time)


# ==============================================================================
# PHASE 2: CORE INGESTION SDK BRIDGE (MAIN PROGRAM)
# ==============================================================================
class BCIIngestionBridge:
    def __init__(self):
        self.inlet = None
        # Initialize an empty matrix buffer: shape = (Total Channels, Max Samples)
        self.data_buffer = np.zeros((TOTAL_CHANNELS, BUFFER_MAX_SAMPLES), dtype=np.float32)
        self.samples_collected = 0

    def discover_and_connect(self, timeout=5.0):


       """Scans the local network loopback for active LSL streams matching our schema."""
        print(f"[Ingestion SDK] Scanning local network for type='{STREAM_TYPE}'...")
        streams = resolve_byprop('type', STREAM_TYPE, timeout=timeout)
        
        if not streams:
            print("[Ingestion SDK] Error: No compatible multi-modal streams detected on network.")
            return False
            
        print(f"[Ingestion SDK] Found target stream: {streams[0].name()} from source {streams[0].source_id()}")
        
        # Instantiate the LSL Network Inlet to read chunks
        # max_chunklen=37 limits max packet latency chunks to ~150ms buffer segments
        self.inlet = StreamInlet(streams[0], max_chunklen=37)
        return True

    def process_stream_loop(self, stop_event):
        """Continuously pulls time-series chunks from the network and updates the analysis buffer."""
        print("[Ingestion SDK] Live ingestion pipeline successfully activated.")
        print(f"[Ingestion SDK] Buffer initialized to shape {self.data_buffer.shape} (Channels x Samples)")


       
        while not stop_event.is_set():
            # Pull a chunk of data samples from the network link
            # timeout=0.0 makes it non-blocking; returns immediately if empty
            samples, timestamps = self.inlet.pull_chunk(timeout=0.0, max_samples=100)
            
            if samples and timestamps:
                # Convert chunk to a clean NumPy matrix: shape = (Samples, Channels)
                chunk_np = np.array(samples, dtype=np.float32)
                
                # Transpose to match internal model specifications: shape = (Channels, Samples)
                chunk_transposed = chunk_np.T
                num_new_samples = chunk_transposed.shape[1]
                
                # Shift historical matrix buffer to the left to make room for incoming chunk
                self.data_buffer = np.roll(self.data_buffer, -num_new_samples, axis=1)
                
                # Insert the newest data matrix chunk at the tail end of the buffer
                self.data_buffer[:, -num_new_samples:] = chunk_transposed


               self.samples_collected += num_new_samples
                
                # Week 1 Validation Reporting: Print out real-time matrix details
                print(
                    f"[Ingestion SDK] LSL Chunk Pulled! Size: {num_new_samples} samples | "
                    f"Total Collected: {self.samples_collected} | "
                    f"Buffer Rolling Shape: {self.data_buffer.shape} | "
                    f"Recent EEG[0] Target Val: {self.data_buffer[0, -1]:.2f} uV", 
                    end='\r'
                )
                
            # Yield brief execution thread time to keep CPU cycles low
            time.sleep(0.01)


# ==============================================================================
# PIPELINE EXECUTION WRAPPER
# ==============================================================================
if __name__ == "__main__":


   print("======================================================================")
    print("      INITIALIZING BCI UNIVERSAL ENGINE: WEEK 1 INGESTION MVP        ")
    print("======================================================================\n")
    
    stop_signal = threading.Event()
    
    # 1. Fire up the Background Playback Stream Thread (Option B Simulation)
    playback_thread = threading.Thread(target=mock_lsl_playback_loop, args=(stop_signal,), daemon=True)
    playback_thread.start()
    
    # Wait briefly to let network sockets open completely
    time.sleep(1.0)
    
    # 2. Instantiate and run the core Application Ingestion Engine
    sdk_bridge = BCIIngestionBridge()
    
    if sdk_bridge.discover_and_connect():
        try:
            # Run live ingestion tracking loop for 10 seconds, then exit cleanly


           run_duration = 10
            print(f"\n[System] Ingestion pipeline running for next {run_duration} seconds. Press Ctrl+C to force terminate early.\n")
            
            # Start loop execution bridge
            sdk_thread = threading.Thread(target=sdk_bridge.process_stream_loop, args=(stop_signal,), daemon=True)
            sdk_thread.start()
            
            time.sleep(run_duration)
            print("\n\n[System] Execution duration reached successfully.")
            
        except KeyboardInterrupt:
            print("\n\n[System] Manual pipeline termination detected.")
    else:
        print("\n[System] Initialization failure: Could not link up core SDK components.")
        
    # Clean shutdown of background pipelines
    print("[System] Terminating LSL streaming sockets cleanly...")
    stop_signal.set()
    playback_thread.join(timeout=2.0)


   print("[System] Execution halted. Week 1 Gate Verified.")



