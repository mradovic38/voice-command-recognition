import pyaudio
import numpy as np

def audio_stream(chunk_size=16000, rate=16000):
    """Continuously capture audio."""
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=rate, input=True, frames_per_buffer=chunk_size)
    print("Listening for commands...")

    try:
        while True:
            audio_chunk = np.frombuffer(stream.read(chunk_size), dtype=np.int16)
            yield audio_chunk
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()