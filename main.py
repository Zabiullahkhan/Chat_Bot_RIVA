
import gradio as gr
# Custom CSS for side tabs and Accenture branding
accenture_css = """
body {
    background-color: #f3f4f6;
    font-family: 'Arial', sans-serif;
    color: #333;
}
h1, h2 {
    color: #0072CE;  /* Accenture Blue */
    text-align: center;
}
.gr-button {
    background-color: #FF6200;  /* Accenture Orange */
    color: white;
    border: none;
    padding: 10px 20px;
    font-size: 16px;
}
.gr-button:hover {
    background-color: #E55D00;
}
.gr-textbox {
    border: 1px solid #0072CE;
    padding: 10px;
    border-radius: 5px;
}
.gr-textbox:focus {
    border-color: #FF6200;
}
.gr-tabs {
    display: flex;
    flex-direction: row;
}
.gr-tab {
    margin-bottom: 10px;
}
.gr-tabpanel {
    flex-grow: 1;
}
"""

# Build the Gradio Interface
with gr.Blocks(css=accenture_css) as demo:
    gr.Markdown("# Welcome to Accenture RAG App with Guardrails")

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Tabs(elem_id="side-tabs"):
                with gr.Tab("Create Embeddings"):
                    url_input = gr.Textbox(label="Enter URL")
                    embed_button = gr.Button("Create Embeddings")
                    embed_output = gr.Textbox(label="Output")

                    # embed_button.click(create_embeddings, inputs=url_input, outputs=embed_output)

                with gr.Tab("Ask Questions"):
                    question_input = gr.Textbox(label="Enter Question")
                    ask_button = gr.Button("Ask Question")
                    answer_output = gr.Textbox(label="Answer")

                    # ask_button.click(ask_question, inputs=question_input, outputs=answer_output)

# Launch the Gradio interface
# demo.launch(server_name="0.0.0.0", server_port=5050)
def fake(message, history):
    if message.strip():
        return gr.Audio("https://github.com/gradio-app/gradio/raw/main/test/test_files/audio_sample.wav")
    else:
        return "Please provide the name of an artist"

def greet(name, intensity):
    return "Hello, " + name + "!" * int(intensity)

import gradio as gr

def synthesize_speech(text, use_knowledge_base, enable_tts_output, asr_language, tts_language, tts_voice):
  """
  This function synthesizes speech from text using the specified settings.

  Args:
      text (str): The text to be synthesized.
      use_knowledge_base (bool): Whether to use the knowledge base.
      enable_tts_output (bool): Whether to enable TTS output.
      asr_language (str): The ASR language.
      tts_language (str): The TTS language.
      tts_voice (str): The TTS voice.

  Returns:
      str: The synthesized speech.
  """
  # Placeholder for actual speech synthesis logic
  speech = f"Synthesized speech for '{text}' with settings: \n"
  speech += f"Use knowledge base: {use_knowledge_base}\n"
  speech += f"Enable TTS output: {enable_tts_output}\n"
  speech += f"ASR language: {asr_language}\n"
  speech += f"TTS language: {tts_language}\n"
  speech += f"TTS voice: {tts_voice}"
  return speech

# Create interface elements
text_input = gr.Textbox(label="Enter text")
use_knowledge_base = gr.Checkbox(label="Use knowledge base")
enable_tts_output = gr.Checkbox(label="Enable TTS output")
asr_language = gr.Dropdown(label="ASR Language", choices=["English (en-US)"], value="English (en-US)")
tts_language = gr.Dropdown(label="TTS Language", choices=["English (en-US)"], value="English (en-US)")
tts_voice = gr.Dropdown(label="TTS Voice", choices=["English-US.Female-1", "English-US.Male-1", "Spanish-ES.Female-1", "Spanish-ES.Male-1"], value="English-US.Female-1")
submit_button = gr.Button(value="Submit")
clear_button = gr.Button(value="Clear")
clear_history_button = gr.Button(value="Clear History")
show_context_button = gr.Button(value="Show Context")
output_text = gr.Textbox(label="Synthesized Speech")

# Create the interface
with gr.Blocks() as demo:
    with gr.Row():
        gr.ChatInterface()
    gr.Interface(
        fn=synthesize_speech,
        inputs=[
            text_input,
            use_knowledge_base,
            enable_tts_output,
            asr_language,
            tts_language,
            tts_voice
        ],
        outputs=output_text,
        title="Text-to-Speech Synthesizer",
        description="Synthesize speech from text with various settings."),
    
    submit_button,
    clear_button, 
    gr.Button(value="clear_history_button"),
    gr.Button(value="show_context_button")

# Launch the interface

demo.launch(debug=True)
