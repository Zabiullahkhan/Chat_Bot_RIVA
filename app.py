"""
Before running client part of Riva, please set up a server. 
The simplest way to do this is to follow quick start guide.
https://docs.nvidia.com/deeplearning/riva/user-guide/docs/quick-start-guide.html#local-deployment-using-quick-start-scripts

"""

#import necesssary 
import riva.client
import riva.client.audio_io
import sys
import requests
import json
import keyboard
import pyaudio
import wave
from utilies.ConfigReader import config
import json

#Authentication
riva_url= config.get('RIVA', 'riva_url')
llm_url = config.get('LLM', 'llm_url')

auth = riva.client.Auth(uri=riva_url)

#setting up services
asr_service = riva.client.ASRService(auth) # for automatic speech recognition
tts_service = riva.client.SpeechSynthesisService(auth) # for text to speech

#set parameters for synthesis service
language_code = 'en-US'
sample_rate_hz = 16000
nchannels = 1
sampwidth = 2

#set parameters for ASR service
from copy import deepcopy
offline_config = riva.client.RecognitionConfig(
    encoding=riva.client.AudioEncoding.LINEAR_PCM,
    max_alternatives=1,
    language_code="en-US",
    enable_automatic_punctuation=True,
    verbatim_transcripts=False,
)
streaming_config = riva.client.StreamingRecognitionConfig(config=deepcopy(offline_config), interim_results=True)


my_wav_file = 'interview-with-bill.wav'
riva.client.add_audio_file_specs_to_config(offline_config, my_wav_file)
riva.client.add_audio_file_specs_to_config(streaming_config, my_wav_file)

#If you intent to use word boosting, then use convenience method riva.client.add_word_boosting_to_config() to add boosting parameters to config.
boosted_lm_words = ['AntiBERTa', 'ABlooper']
boosted_lm_score = 20.0
riva.client.add_word_boosting_to_config(offline_config, boosted_lm_words, boosted_lm_score)
riva.client.add_word_boosting_to_config(streaming_config, boosted_lm_words, boosted_lm_score)

#print(offline_config)
#print(streaming_config)

# offline recognition with audio file
# output_device = None  # use default device
# wav_parameters = riva.client.get_wav_file_parameters(my_wav_file)
# sound_callback = riva.client.audio_io.SoundCallBack(
#     output_device, wav_parameters['sampwidth'], wav_parameters['nchannels'], wav_parameters['framerate'],
# )
# audio_chunk_iterator = riva.client.AudioChunkFileIterator(my_wav_file, 4800, sound_callback)
# response_generator = asr_service.streaming_response_generator(audio_chunk_iterator, streaming_config)
# riva.client.print_streaming(response_generator, show_intermediate=True)
# sound_callback.close()


# streaming with microphone

def asr_with_microphone():
    """
    streaming audio with microphone converte to text and save to a file.
    """
    output_file = "Output/my_results.txt"
    input_device = None  # default device
    print("\n\nTap space when you're ready. ", end="", flush=True)
    keyboard.wait('space')

    with riva.client.audio_io.MicrophoneStream(
        rate=streaming_config.config.sample_rate_hertz,
            chunk=streaming_config.config.sample_rate_hertz // 10,
            device=input_device) as audio_chunk_iterator:
         riva.client.print_streaming(
                  responses = next(asr_service.streaming_response_generator(
                        audio_chunks=audio_chunk_iterator,
                        streaming_config=streaming_config,
                    )),
                show_intermediate=False,
                output_file=[sys.stdout, output_file],
            )
        
    print("\n\nStreaming stopped.")


    # Define the API URL
    #api_url = "http://127.0.0.1:8890/generate"


# prompt = "What is new Delhi"

payload = {
    "messages":[
        # {"role": "user", "content": prompt},
        {"role": "user", "content": "Hello there how are you?"},
        # {"role": "assistant", "content": "Good and you?"}, 
        # {"role": "user", "content": "What is the capital of India?"}
    ],
    
    "model": "mistral-7b-instruct",
    # "use_knowledge_base": False,
    "temperature": 0.2,
    "n" : 1,
    "top_p": 0.7,
    "max_tokens": 1024,
    "stop" : ["string"]
}

# payload = {
#     "model": model_name,
#     "messages": messages
# }


# payload.update(max_tokens=1024)


def get_llm_response(url,text=None):
    try:
        strOutcome = "Default Failure:call to function get_llm_response failed"
        headers = {
                "Accept": "application/json",
                "Content-Type": "application/json"}
        
        with open("config/llm_payload.json", "r") as file:
            payload = json.load(file)
        if text:
            payload['messages'][2]["content"] = text

        response = requests.post(url, headers=headers, data=json.dumps(payload))
        # print(response.json()['choices'][0]['message']['content'])
        if response.status_code == 200:
            # return response.json()
            res= response.json()['choices'][0]['message']['content']
        else:
            res =f"Failed to get response: {response.status_code} - {response.text}"
            
    
    except :
        strOutcome = f"Request failed with exception: {sys.exc_info()}"
    else:
        strOutcome = res
    finally:
            return strOutcome
        

def synthesize_online(response):
    print("\nsophia:",response)
    output_device = None  # use default device
    sound_stream = riva.client.audio_io.SoundCallBack(
        output_device, nchannels=nchannels, sampwidth=sampwidth, framerate=sample_rate_hz
    )
    for resp in tts_service.synthesize_online(response, language_code=language_code, sample_rate_hz=sample_rate_hz):
        sound_stream(resp.audio)


def asr_offline(output_audio_file):
    print("you:",end="")
    output_text_file = "Output/my_results.txt"
    audio_chunk_iterator = riva.client.AudioChunkFileIterator(output_audio_file, 4800, riva.client.sleep_audio_length)
    response_generator = asr_service.streaming_response_generator(audio_chunk_iterator, streaming_config)
    riva.client.print_streaming(response_generator, show_intermediate=False,output_file=[sys.stdout, output_text_file])


def record_audio():
    """
    record audio from microphone and transcribe it.
    """
    output_audio_file = "Output/voice_record.wav"
    
    # Wait until user presses space bar
    print("\n\nTap space when you're ready. ", end="", flush=True)
    # keyboard.wait('space')
    while keyboard.is_pressed('space'): pass

    # Record from microphone until user presses space bar again
    print("I'm all ears. Tap space when you're done.\n")
    audio, frames = pyaudio.PyAudio(), []
    py_stream = audio.open(rate=16000, format=pyaudio.paInt16, channels=1, input=True, frames_per_buffer=512)
    while not keyboard.is_pressed('space'): 
        frames.append(py_stream.read(512))
    py_stream.stop_stream(), py_stream.close(), audio.terminate()


    # save recording 
    with wave.open(output_audio_file, 'wb') as wf:
        wf.setparams((1, audio.get_sample_size(pyaudio.paInt16), 16000, 0, 'NONE', 'NONE'))
        wf.writeframes(b''.join(frames))

    #transcribe recording
    asr_offline(output_audio_file)




if __name__ == "__main__":
    # asr_with_microphone()
    
    while True:
        print("\npress 'enter' to exit the conversation or Tap 'space' to continue...")
        keyboard.read_key()
        if keyboard.is_pressed('enter'):
            break
        
        record_audio()
        with open("Output/my_results.txt", "r") as file:
            text = file.read().replace("##", "")
        
        
            # response = get_llm_response(llm_url, {"content": text})
            # synthesize_online(response
        res=get_llm_response(llm_url,text)                         
        synthesize_online(res)
                                                    
