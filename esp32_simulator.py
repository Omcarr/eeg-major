"""
ESP32 EEG Data Simulator
Simulates ESP32 sending real-time EEG data to FastAPI WebSocket server
Use this for testing before connecting actual hardware
"""

import asyncio
import websockets
import json
import numpy as np
from datetime import datetimex
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ESP32Simulator:
    """Simulates ESP32 device sending EEG data"""
    
    def __init__(self, server_url="ws://localhost:8000/ws/eeg"):
        self.server_url = server_url
        self.channels = ['Pz', 'P3', 'P4', 'O1', 'O2', 'Cz']
        self.sampling_rate = 128  # Hz
        self.is_running = False
        self.time_offset = 0
        
    def generate_eeg_sample(self, channel, time_sec):
        """
        Generate realistic EEG signal sample
        
        EEG bands:
        - Delta (0.5-4 Hz): Deep sleep
        - Theta (4-8 Hz): Drowsiness, meditation
        - Alpha (8-13 Hz): Relaxed, eyes closed
        - Beta (13-30 Hz): Active thinking, focus
        - Gamma (30-50 Hz): High-level information processing
        """
        
        # Different channels have slightly different characteristics
        channel_offsets = {
            'Pz': 0.0,
            'P3': 0.1,
            'P4': 0.2,
            'O1': 0.3,
            'O2': 0.4,
            'Cz': 0.5
        }
        
        t = time_sec + channel_offsets.get(channel, 0)
        
        # Delta wave (0.5-4 Hz) - dominant in deep sleep
        delta = 3.0 * np.sin(2 * np.pi * 2.0 * t)
        
        # Theta wave (4-8 Hz) - drowsiness
        theta = 2.0 * np.sin(2 * np.pi * 6.0 * t)
        
        # Alpha wave (8-13 Hz) - relaxed, eyes closed (dominant in healthy awake)
        alpha = 5.0 * np.sin(2 * np.pi * 10.0 * t)
        
        # Beta wave (13-30 Hz) - active thinking
        beta = 2.5 * np.sin(2 * np.pi * 20.0 * t)
        
        # Gamma wave (30-50 Hz) - high-level processing
        gamma = 1.0 * np.sin(2 * np.pi * 40.0 * t)
        
        # Add realistic noise
        noise = np.random.normal(0, 1.0)
        
        # Occasional artifacts (eye blinks, muscle movements)
        if np.random.random() < 0.01:  # 1% chance
            noise += np.random.normal(0, 10.0)
        
        # Combine all components
        sample = delta + theta + alpha + beta + gamma + noise
        
        # Add slight channel-specific variations
        channel_factor = 1.0 + (hash(channel) % 20 - 10) / 100.0
        sample *= channel_factor
        
        return round(sample, 4)
    
    async def send_single_samples(self, websocket):
        """Send one sample per channel at a time (streaming mode)"""
        logger.info("Starting single-sample streaming mode...")
        logger.info(f"Sampling rate: {self.sampling_rate} Hz")
        
        sample_interval = 1.0 / self.sampling_rate  # Time between samples
        sample_count = 0
        
        try:
            while self.is_running:
                start_time = asyncio.get_event_loop().time()
                
                # Generate and send one sample for each channel
                current_time = self.time_offset + sample_count / self.sampling_rate
                
                for channel in self.channels:
                    value = self.generate_eeg_sample(channel, current_time)
                    
                    message = {
                        'type': 'sample',
                        'channel': channel,
                        'value': value,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    await websocket.send(json.dumps(message))
                
                sample_count += 1
                
                # Log progress every second
                if sample_count % self.sampling_rate == 0:
                    logger.info(f"Sent {sample_count} samples ({sample_count // self.sampling_rate}s of data)")
                
                # Wait for next sample time
                elapsed = asyncio.get_event_loop().time() - start_time
                sleep_time = max(0, sample_interval - elapsed)
                await asyncio.sleep(sleep_time)
                
        except Exception as e:
            logger.error(f"Error in streaming: {e}")
    
    async def send_batch_samples(self, websocket, batch_size=128):
        """Send batch of samples at once (batch mode)"""
        logger.info(f"Starting batch mode (batch size: {batch_size})...")
        
        batch_count = 0
        
        try:
            while self.is_running:
                # Generate batch for all channels
                batch_data = {}
                
                for channel in self.channels:
                    samples = []
                    for i in range(batch_size):
                        t = self.time_offset + (batch_count * batch_size + i) / self.sampling_rate
                        value = self.generate_eeg_sample(channel, t)
                        samples.append(value)
                    
                    batch_data[channel] = samples
                
                # Send batch
                message = {
                    'type': 'batch',
                    'data': batch_data,
                    'timestamp': datetime.now().isoformat()
                }
                
                await websocket.send(json.dumps(message))
                
                batch_count += 1
                total_samples = batch_count * batch_size
                logger.info(f"Sent batch {batch_count} ({total_samples} total samples, {total_samples / self.sampling_rate:.1f}s)")
                
                # Wait before next batch (simulate real-time transmission)
                await asyncio.sleep(batch_size / self.sampling_rate)
                
        except Exception as e:
            logger.error(f"Error in batch mode: {e}")
    
    async def run(self, mode='single', duration=None, auto_predict=True):
        """
        Run the simulator
        
        Args:
            mode: 'single' for sample-by-sample, 'batch' for batch transmission
            duration: How long to run (seconds), None for infinite
            auto_predict: Automatically request prediction when enough data collected
        """
        logger.info("="*60)
        logger.info("ESP32 EEG SIMULATOR")
        logger.info("="*60)
        logger.info(f"Server: {self.server_url}")
        logger.info(f"Mode: {mode}")
        logger.info(f"Channels: {', '.join(self.channels)}")
        logger.info("="*60)
        
        try:
            async with websockets.connect(self.server_url) as websocket:
                logger.info("✓ Connected to server")
                self.is_running = True
                
                # Start sending data based on mode
                if mode == 'single':
                    task = asyncio.create_task(self.send_single_samples(websocket))
                elif mode == 'batch':
                    task = asyncio.create_task(self.send_batch_samples(websocket))
                else:
                    raise ValueError(f"Unknown mode: {mode}")
                
                # Auto-predict after collecting enough data
                if auto_predict:
                    # Wait for 1024 samples to be collected (8 seconds at 128 Hz)
                    await asyncio.sleep(10)
                    
                    logger.info("\n" + "="*60)
                    logger.info("REQUESTING PREDICTION")
                    logger.info("="*60)
                    
                    await websocket.send(json.dumps({'type': 'predict'}))
                    
                    # Wait for response
                    response = await websocket.recv()
                    result = json.loads(response)
                    
                    if result.get('type') == 'prediction':
                        self.display_prediction(result['result'])
                    elif result.get('type') == 'error':
                        logger.error(f"Prediction error: {result.get('message')}")
                
                # Run for specified duration or until stopped
                if duration:
                    await asyncio.sleep(duration - 10)  # Subtract time already waited
                    self.is_running = False
                    await task
                else:
                    # Run indefinitely
                    await task
                
        except websockets.exceptions.ConnectionClosed:
            logger.error("Connection closed by server")
        except Exception as e:
            logger.error(f"Error: {e}")
        finally:
            self.is_running = False
            logger.info("Simulator stopped")
    
    def display_prediction(self, result):
        """Display prediction results"""
        logger.info("\n" + "="*60)
        logger.info("PREDICTION RESULT")
        logger.info("="*60)
        logger.info(f"Diagnosis: {result['prediction']}")
        logger.info(f"Confidence: {result['confidence']*100:.2f}%")
        logger.info(f"\nDetailed Probabilities:")
        logger.info(f"  Healthy: {result['healthy_probability']*100:.2f}%")
        logger.info(f"  Alzheimer's: {result['alzheimer_probability']*100:.2f}%")
        logger.info(f"\nTimestamp: {result['timestamp']}")
        logger.info("="*60 + "\n")

# ============================================
# COMMAND LINE INTERFACE
# ============================================

async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ESP32 EEG Data Simulator')
    parser.add_argument('--server', default='ws://localhost:8000/ws/eeg',
                       help='WebSocket server URL')
    parser.add_argument('--mode', choices=['single', 'batch'], default='single',
                       help='Transmission mode: single samples or batches')
    parser.add_argument('--duration', type=int, default=None,
                       help='Duration in seconds (default: infinite)')
    parser.add_argument('--no-predict', action='store_true',
                       help='Disable automatic prediction')
    
    args = parser.parse_args()
    
    simulator = ESP32Simulator(server_url=args.server)
    
    try:
        await simulator.run(
            mode=args.mode,
            duration=args.duration,
            auto_predict=not args.no_predict
        )
    except KeyboardInterrupt:
        logger.info("\nStopped by user")

# ============================================
# QUICK START FUNCTIONS
# ============================================

async def quick_test():
    """Quick 30-second test with prediction"""
    logger.info("🚀 Running 30-second quick test...")
    simulator = ESP32Simulator()
    await simulator.run(mode='single', duration=30, auto_predict=True)

async def continuous_stream():
    """Continuous streaming until stopped"""
    logger.info("🌊 Starting continuous stream (Ctrl+C to stop)...")
    simulator = ESP32Simulator()
    await simulator.run(mode='single', duration=None, auto_predict=False)

async def batch_test():
    """Test batch transmission mode"""
    logger.info("📦 Testing batch transmission mode...")
    simulator = ESP32Simulator()
    await simulator.run(mode='batch', duration=30, auto_predict=True)

# ============================================
# RUN
# ============================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'quick':
            asyncio.run(quick_test())
        elif sys.argv[1] == 'stream':
            asyncio.run(continuous_stream())
        elif sys.argv[1] == 'batch':
            asyncio.run(batch_test())
        else:
            asyncio.run(main())
    else:
        # Default: run with arguments
        asyncio.run(main())

# ============================================
# USAGE EXAMPLES
# ============================================
"""
Usage Examples:

1. Quick 30-second test:
   python esp32_simulator.py quick

2. Continuous streaming:
   python esp32_simulator.py stream

3. Batch mode test:
   python esp32_simulator.py batch

4. Custom configuration:
   python esp32_simulator.py --mode single --duration 60
   python esp32_simulator.py --server ws://192.168.1.100:8000/ws/eeg
   python esp32_simulator.py --mode batch --no-predict

5. Default (single mode with auto-predict):
   python esp32_simulator.py
"""