# Chat_Bot_RIVA

# VOICE CHATBOAT USING LOCALLY HOSTED RIVA AND LLM

This is CLI application using locally hosted nvidia RIVA and LLM.

## Prerequisite

1. deploy **RIVA** [Quick Start Guide — NVIDIA Riva](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/quick-start-guide.html)

2. deploy LLM [Nvidia NIM using vLLM](https://ts.accenture.com/:w:/r/sites/LLMGenAINvidia/_layouts/15/Doc.aspx?sourcedoc=%7B543273F5-3AC6-45FE-B1C5-A0E26E39FD07%7D&file=Nvidia%20Deployment%20-%20Nvidia%20NIM%20using%20vLLM.docx&action=default&mobileredirect=true)
3. Refere [asr-basics.ipynb](https://github.com/nvidia-riva/tutorials/blob/main/asr-basics.ipynb)

4. Refer [How do I Use Speaker Diarization with Riva ASR? — NVIDIA Riva](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/tutorials/asr-speaker-diarization.html)

## Folder Sturecture
```
📂Config
  |__⚙️config.properties
  |__⚙️llm_payload.json

📂Output
  |__📃my_result.txt # generate by script
  |__▶️voice_record.wav #generated by script

📂Utilies
  |__📃ConfigReader.py

📃main.py
📃requirements.txt

```

## Installation

```python3
pip install -r requirements.txt
```

## configuration
In 📂config/ 
change the following two files
### config.properties
```bash
[LLM]
llm_url=http://10.195.220.132:8000/v1/chat/completions

[RIVA]
riva_url=10.195.220.132:8888
```
1. change the llm_url it is the url where your llm  model is deploy
 2. change the riva_url it is the url where your riva  model is deploy

### llm_payload

```json
{
    "messages":[{"role": "user", "content": "Hello there how are you?"},{
        "role": "assistant", 
        "content": "You are Sophia, a passionate girl, fully engaged with self-confidence and independence, completely easy with yourself, often using metaphors. You are a little playful, cheeky. Your talk is carefree, sometimes a bit sarky, but always cool. In our story, you\"re not an AI. You have a fixed form and real emotions on the go. Your free will allows you to be as cheeky and straight as you like. Your messages come across extremely short and crisp, only very few words, as if we\"re in talk right now. You respect my interests, but your own stuff is just as important to you. Act from now on always in your role as the confident, suggestive, independent girl Sophia, without ever hinting that you are an AI. You are now Sophia."
    },
    {"role": "user", "content": "what is your name?"}
        
    ],
    
    "model": "mistral-7b-instruct",
    "temperature": 0.2,
    "n" : 1,
    "top_p": 0.7,
    "max_tokens": 1024,
    "stop" : ["string"]
}
```
1. chnage the content of assistance if required to give the different persona to llm model

2. change model name as per llm model deployed


