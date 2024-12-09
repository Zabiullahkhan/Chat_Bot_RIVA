import pyaudio
import keyboard

while True:
    # Wait until user presses space bar
    print("\n\nTap space when you're ready. ", end="", flush=True)
    keyboard.wait('space')
    while keyboard.is_pressed('space'): pass

    # Record from microphone until user presses space bar again
    print("I'm all ears. Tap space when you're done.\n")
    audio, frames = pyaudio.PyAudio(), []
    py_stream = audio.open(rate=16000, format=pyaudio.paInt16, channels=1, input=True, frames_per_buffer=512)
    while not keyboard.is_pressed('space'): 
        frames.append(py_stream.read(512))
    py_stream.stop_stream(), py_stream.close(), audio.terminate()

    
