import time
from .kokoro_tts import KokoroTTS

class TTSService:
    def __init__(self):
        self.tts = None
        self.load_model()
    
    def load_model(self):
        """Load the model once and keep it in memory"""
        print("Loading TTS model...")
        load_start = time.time()
        self.tts = KokoroTTS()
        load_time = time.time() - load_start
        print(f"Model loaded in {load_time:.2f} seconds")
    
    def speak(self, text, voice='af_heart'):
        """Generate speech with timing info"""
        if not self.tts:
            raise RuntimeError("TTS model not loaded")
        
        generation_start = time.time()
        audio, sr, _ = self.tts.synthesize(text, voice=voice)
        generation_time = time.time() - generation_start
        
        print(f"TTS generation time: {generation_time:.3f} seconds")
        print(f"Audio length: {len(audio)} samples at {sr} Hz")
        print(f"Audio duration: {len(audio) / sr:.3f} seconds")
        
        return audio, sr, generation_time

def interactive_tts():
    """Interactive TTS session"""
    service = TTSService()
    
    print("\n=== Interactive TTS Session ===")
    print("Type text to synthesize (or 'quit' to exit)")
    
    while True:
        text = input("\nEnter text: ").strip()
        
        if text.lower() in ['quit', 'exit', 'q']:
            break
        
        if not text:
            continue
        
        try:
            audio, sr, gen_time = service.speak(text)
            print(f"✓ Generated {len(audio)} samples in {gen_time:.3f}s")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    interactive_tts()