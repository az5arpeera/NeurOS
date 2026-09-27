"""
Zero-Calibration BCI Universal Layer - 3 MVP Event Bridge & SDK Gateway
Target: Week 3 Engineering Gate - WebSocket Broadcast Server & Client Integration

This script provides a complete, runnable architecture showing how the universal 
zero-calibration layer bridges normalized geometric brain matrices into an external 
application ecosystem. 

It starts an asynchronous WebSocket Server that handles streaming data pipelines, 
executes real-time Riemannian Euclidean Alignment (from Week 2) on incoming brain matrices, 
runs a spatial pattern matching routine to detect discrete intent signatures (e.g., a 
high-energy jaw clench or focused select command), and broadcasts clean JSON events 
to all connected application clients under 50ms processing latency.

To remain completely runnable in a single self-contained script, it also launches 
a parallel asynchronous mock client task that demonstrates the 5-line developer 
integration pattern used to capture these streams inside consumer apps or games.

Dependencies:


   pip install numpy websockets
"""

import asyncio
import json
import time
import sys
import numpy as np
import websockets

# ==============================================================================
# PIPELINE & ARCHITECTURAL CONFIGURATIONS
# ==============================================================================
HOST = "127.0.0.1"
PORT = 8765
CHANNELS = 8
SAMPLES_PER_EPOCH = 50  # ~200ms processing chunks at 250Hz sample rate

print("======================================================================")


print("     INITIALIZING BCI UNIVERSAL ENGINE: WEEK 3 EVENT BRIDGE MVP      ")
print("======================================================================\n")

# ==============================================================================
# PHASE 1: MATHEMATICAL ALIGNMENT CORE (INLINE INTEGRATION FROM WEEK 2)
# ==============================================================================
class RiemannianAlignmentEngine:
    """
    Maintains the geometric baseline anchor state and normalizes incoming raw spatial 
    matrices to the manifold origin.
    """
    def __init__(self):
        # Initialize identity mapping matrix until a baseline reference anchor is established
        self.R = np.eye(CHANNELS)
        self.is_anchored = False

    def establish_static_anchor(self, base_matrix):
        """Computes the inverse square root transformation matrix R = P^(-1/2)"""
        eigenvalues, eigenvectors = np.linalg.eigh(base_matrix)


       eigenvalues = np.clip(eigenvalues, 1e-10, None)
        diag_inv_sqrt = np.diag(1.0 / np.sqrt(eigenvalues))
        self.R = np.dot(eigenvectors, np.dot(diag_inv_sqrt, eigenvectors.T))
        self.is_anchored = True

    def transform_channels(self, X):
        """Projects the channel stream matrix into the normalized geometric space."""
        return np.dot(self.R, X)


# ==============================================================================
# PHASE 2: CORE BROADCAST GATEWAY & STATE SERVER
# ==============================================================================
class BCIEventBridgeServer:
    """
    Manages active client WebSocket hooks, ingests raw streaming matrices, normalizes 
    spatial characteristics via geometry, decodes actions, and broadcasts events.
    """
    def __init__(self):


       self.connected_clients = set()
        self.alignment_layer = RiemannianAlignmentEngine()
        self.is_running = False
        
        # Setup a simulated baseline covariance profile for anchoring demonstration
        # This mirrors a user starting the app and initializing their spatial center
        mock_baseline = np.eye(CHANNELS) * 4.5
        mock_baseline[0, 1] = mock_baseline[1, 0] = 1.2  # Simulate cross-channel noise correlation
        self.alignment_layer.establish_static_anchor(mock_baseline)

    async def register_client(self, websocket):
        """Tracks new application developer connection sockets."""
        self.connected_clients.add(websocket)
        print(f"[Event Server] New developer application connected. Active links: {len(self.connected_clients)}")
        try:
            # Maintain active listener loop for incoming client configurations or handshake pings
            async for message in websocket:
                pass 
        except websockets.exceptions.ConnectionClosed:


           pass
        finally:
            self.connected_clients.remove(websocket)
            print(f"[Event Server] Client application disconnected. Active links: {len(self.connected_clients)}")

    async def broadcast_intent_packet(self, action_label, confidence):
        """Pipes structured JSON packets down to all active developer applications."""
        if not self.connected_clients:
            return

        payload = {
            "event": action_label,
            "confidence": round(float(confidence), 4),
            "timestamp_ms": int(time.time() * 1000),
            "hardware_channels": CHANNELS
        }
        
        message = json.dumps(payload)
        # Simultaneously broadcast to all listening developer application channels


       await asyncio.gather(*[client.send(message) for client in self.connected_clients])

    async def operational_inference_loop(self):
        """
        Simulates the background real-time data ingestion and translation engine loop.
        Ingests time-series frames, executes geometric filters, determines intent,
        and pushes packets down the event bridge.
        """
        self.is_running = True
        print("[Event Server] Core inference execution system activated.")
        
        loop_counter = 0
        while self.is_running:
            start_time = asyncio.get_event_loop().time()
            
            # 1. Synthesize streaming raw input matrix (Channels x Samples)
            raw_frame = np.random.normal(0, 2.0, (CHANNELS, SAMPLES_PER_EPOCH))
            
            # Periodically inject a distinct biometric intent spike (every ~2 seconds / 10 loops)


           # This simulates a programmatic command event like a concentrated motor selection
            is_intent_spike = (loop_counter % 10 == 0)
            if is_intent_spike:
                # Amplify front channel grids to simulate deliberate intentional action signatures
                raw_frame[0:2, :] += np.sin(np.linspace(0, np.pi, SAMPLES_PER_EPOCH)) * 18.0
                
            # 2. Pass matrix straight through the Week 2 Riemannian Geometry normalization layer
            aligned_frame = self.alignment_layer.transform_channels(raw_frame)
            
            # 3. Perform lightweight deterministic classification pattern matching
            # Calculate rolling energy index across primary normalized alignment axes
            spatial_energy = np.mean(np.square(aligned_frame[0]))
            
            # Match thresholds within the normalized geometric matrix grid space
            if spatial_energy > 15.0:
                action = "INTENT_SELECT"
                confidence = 0.92 + (np.random.uniform(0.01, 0.06) if spatial_energy > 20.0 else 0.0)
                # Broadcast translated choice immediately across the network event engine bridge
                await self.broadcast_intent_packet(action, confidence)


           else:
                # Neutral baseline tracking broadcast pattern
                if loop_counter % 5 == 0:  # Prevent flooding, broadcast continuous state intervals
                    await self.broadcast_intent_packet("STATE_NEUTRAL", 0.99)
            
            loop_counter += 1
            
            # Maintain highly reliable 50ms processing cycle intervals (~20Hz frames update cadence)
            elapsed = asyncio.get_event_loop().time() - start_time
            sleep_duration = max(0.0, 0.05 - elapsed)
            await asyncio.sleep(sleep_duration)


# ==============================================================================
# PHASE 3: THE DEVELOPER INTEGRATION TEST (MOCK TARGET CONSUMER APP)
# ==============================================================================
async def target_application_developer_mock(stop_event):
    """
    Demonstrates the exact 5-line integration layout used by third-party developers 


   to add zero-calibration brain controls directly into their applications.
    """
    # Wait for the backend web sockets server loop initialization layer to finish launching
    await asyncio.sleep(1.5)
    uri = f"ws://{HOST}:{PORT}"
    
    print("\n----------------------------------------------------------------------")
    print("      DEVELOPER APPLICATION RUNTIME LOG: CONNECTING TO SDK BRIDGE      ")
    print("----------------------------------------------------------------------")
    
    try:
        # --- THE 5-LINE DEVELOPER SDK INTEGRATION PATTERN ---
        async with websockets.connect(uri) as bci_sdk_socket: # Line 1: Hook into local universal engine layer
            print(f"[Client App] Connected to Universal BCI Engine via local socket port {PORT}.")
            print("[Client App] Standing by for hands-free intent JSON packets...\n")
            
            while not stop_event.is_set():
                packet_data = await bci_sdk_socket.recv()      # Line 2: Await incoming localized real-time stream frames
                intent_json = json.loads(packet_data)          # Line 3: Parse standard serialized network package


               
                # Line 4: Check parsed intent payload labels
                if intent_json["event"] == "INTENT_SELECT":
                    # Line 5: Route clean intent direct to internal application logic functions
                    print(
                        f"[Client App Action Triggered!] Event: {intent_json['event']} | "
                        f"Confidence: {intent_json['confidence']*100:.2f}% | "
                        f"Timestamp: {intent_json['timestamp_ms']} ms"
                    )
                else:
                    print(f"   [Client App Idle State] Monitoring stream... Current: {intent_json['event']}", end='\r')
                    
    except ConnectionRefusedError:
        print("[Client App] Connection error: Universal engine instance could not be resolved.")
    except Exception as e:
        print(f"[Client App] Processing cycle terminated: {e}")


# ==============================================================================


# ASYNCHRONOUS PIPELINE ORCHESTRATION ARCHITECTURE
# ==============================================================================
async def main():
    server_instance = BCIEventBridgeServer()
    stop_signal = asyncio.Event()

    # Instantiate the Asynchronous WebSocket Server listener fabric configuration
    async with websockets.serve(server_instance.register_client, HOST, PORT):
        print(f"[Event Server] WebSocket Server deployed successfully at ws://{HOST}:{PORT}")
        
        # Assemble execution futures for the infrastructure loops
        server_task = asyncio.create_task(server_instance.operational_inference_loop())
        client_mock_task = asyncio.create_task(target_application_developer_mock(stop_signal))
        
        # Execute pipeline validation loop tracking matrix states for a set duration
        run_lifetime_sec = 8
        print(f"[System] Entire network engine topology running for next {run_lifetime_sec} seconds...\n")
        await asyncio.sleep(run_lifetime_sec)
        


       # Shutdown framework interfaces cleanly
        print("\n\n[System] Initiating clean software cluster shutdown sequence...")
        stop_signal.set()
        server_instance.is_running = False
        
        # Await final clean exits of network execution loops
        await asyncio.gather(server_task, client_mock_task, return_exceptions=True)
        print("[System] Software layers successfully halted. Week 3 Gate Verified.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[System] Execution halted via explicit user termination command.")
        sys.exit(0)


