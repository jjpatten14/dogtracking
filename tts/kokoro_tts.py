import soundfile as sf
import numpy as np
import time
import os
import torch
from kokoro import KPipeline

class KokoroTTS:
    def __init__(self):
        # Check for CUDA availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        if torch.cuda.is_available():
            print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        print("Loading Kokoro-82M model...")
        self.pipeline = KPipeline(lang_code='a', device=self.device, repo_id='hexgrad/Kokoro-82M')
        print("Model loaded successfully!")
    
    def synthesize(self, text, output_path="output.wav", voice='af_heart'):
        """
        Synthesize speech from text
        
        Args:
            text (str): Input text to synthesize
            output_path (str): Path to save the audio file
            voice (str): Voice to use for synthesis
        
        Returns:
            tuple: (audio_array, sample_rate, synthesis_time)
        """
        start_time = time.time()
        
        print(f"Synthesizing with Kokoro: '{text}'")
        
        # Generate speech using Kokoro
        generator = self.pipeline(text, voice=voice)
        
        # Get the audio from generator
        audio_chunks = []
        for i, (gs, ps, audio) in enumerate(generator):
            audio_chunks.append(audio)
        
        # Combine all audio chunks
        if audio_chunks:
            audio = np.concatenate(audio_chunks)
        else:
            audio = np.array([])
        
        sample_rate = 24000  # Kokoro uses 24kHz
        
        # Save audio file if path provided
        if output_path:
            sf.write(output_path, audio, sample_rate)
        
        synthesis_time = time.time() - start_time
        
        return audio, sample_rate, synthesis_time
    
    def stream_synthesize(self, text_stream, chunk_size=50, voice='af_heart'):
        """
        Stream synthesis for real-time applications
        
        Args:
            text_stream (iter): Iterator of text chunks
            chunk_size (int): Size of text chunks to process
            voice (str): Voice to use for synthesis
        
        Yields:
            tuple: (audio_chunk, sample_rate)
        """
        buffer = ""
        
        for chunk in text_stream:
            buffer += chunk
            
            # Process when buffer reaches chunk size or contains sentence end
            if len(buffer) >= chunk_size or any(punct in buffer for punct in '.!?'):
                if buffer.strip():
                    audio, sr, _ = self.synthesize(buffer.strip(), output_path=None, voice=voice)
                    yield audio, sr
                    buffer = ""
        
        # Process remaining buffer
        if buffer.strip():
            audio, sr, _ = self.synthesize(buffer.strip(), output_path=None, voice=voice)
            yield audio, sr

def test_kokoro():
    """Test the Kokoro TTS implementation"""
    try:
        tts = KokoroTTS()
        
        # Test synthesis
        test_text = "Hello, this is a test of the Kokoro TTS system for home assistant."
        print(f"Synthesizing: '{test_text}'")
        
        audio, sr, synthesis_time = tts.synthesize(test_text, "test_output.wav")
        
        print(f"Synthesis completed in {synthesis_time:.3f} seconds")
        print(f"Audio length: {len(audio)} samples at {sr} Hz")
        print(f"Audio duration: {len(audio) / sr:.3f} seconds")
        
        return True
        
    except Exception as e:
        print(f"Error testing Kokoro TTS: {e}")
        return False

if __name__ == "__main__":
    test_kokoro()