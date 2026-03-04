import gradio as gr
import os
import soundfile as sf
from kanade_tokenizer import load_audio
from core.cloner import KokoClone
from core.chunked_convert import chunked_voice_conversion

# 1. Initialize the cloner globally so models load only once when the server starts
print("Loading KokoClone models for the Web UI...")
cloner = KokoClone()
def clone_voice(text, lang, ref_audio_path):
    """Gradio handler: text + reference audio → cloned speech."""
    if not text or not text.strip():
        raise gr.Error("Please enter some text.")
    if not ref_audio_path:
        raise gr.Error("Please upload or record a reference audio file.")

    output_file = "gradio_output.wav"

    try:
        cloner.generate(
            text=text,
            lang=lang,
            reference_audio=ref_audio_path,
            output_path=output_file
        )
        return output_file
    except Exception as e:
        raise gr.Error(f"An error occurred during generation: {str(e)}")


def convert_voice(source_audio_path, ref_audio_path):
    """Gradio handler: source audio + reference audio → re-voiced speech."""
    if not source_audio_path:
        raise gr.Error("Please upload or record a source audio file.")
    if not ref_audio_path:
        raise gr.Error("Please upload or record a reference audio file.")

    output_file = "gradio_convert_output.wav"

    try:
        # Load both audio files onto the same device as the Kanade model
        source_wav = load_audio(source_audio_path, sample_rate=cloner.sample_rate).to(cloner.device)
        ref_wav = load_audio(ref_audio_path, sample_rate=cloner.sample_rate).to(cloner.device)

        converted = chunked_voice_conversion(
            kanade=cloner.kanade,
            vocoder_model=cloner.vocoder,
            source_wav=source_wav,
            ref_wav=ref_wav,
            sample_rate=cloner.sample_rate,
        )

        sf.write(output_file, converted.numpy(), cloner.sample_rate)
        return output_file
    except Exception as e:
        raise gr.Error(f"An error occurred during conversion: {str(e)}")

# 2. Build the Gradio UI using Blocks
with gr.Blocks() as demo:
    gr.Markdown(
        """
        <div style="text-align: center;">
            <h1>🎧 KokoClone</h1>
            <p>Voice Cloning, Now Inside Kokoro.<br>
            Generate natural multilingual speech and clone any target voice with ease.<br>
            <i>Built on Kokoro TTS.</i></p>
        </div>
        """
    )

    with gr.Tabs():
        # ── Tab 1: Text → Cloned Speech ─────────────────────────────────────
        with gr.Tab("🎤 Text → Clone"):
            with gr.Row():
                with gr.Column(scale=1):
                    text_input = gr.Textbox(
                        label="1. Text to Synthesize",
                        lines=4,
                        placeholder="Enter the text you want spoken..."
                    )

                    lang_input = gr.Dropdown(
                        label="2. Language",
                        choices=[
                            ("English", "en"),
                            ("Hindi", "hi"),
                            ("French", "fr"),
                            ("Japanese", "ja"),
                            ("Chinese", "zh"),
                            ("Italian", "it"),
                            ("Spanish", "es"),
                            ("Portuguese", "pt")
                        ],
                        value="en"
                    )

                    ref_audio_input = gr.Audio(
                        label="3. Reference Voice (Upload or Record)",
                        type="filepath"
                    )

                    submit_btn = gr.Button("🚀 Generate Clone", variant="primary")

                with gr.Column(scale=1):
                    output_audio = gr.Audio(
                        label="Generated Cloned Audio",
                        interactive=False,
                        autoplay=False
                    )

                    gr.Markdown(
                        """
                        <br>

                        ### 💡 Tips for Best Results:
                        * **Clean Audio:** Use a reference audio clip without background noise or music.
                        * **Length:** A reference clip of 3 to 10 seconds is usually the sweet spot.
                        * **Language Match:** Make sure the selected language matches the text you typed!
                        * **First Run:** The very first generation might take a few extra seconds while the models allocate memory.
                        """
                    )

            submit_btn.click(
                fn=clone_voice,
                inputs=[text_input, lang_input, ref_audio_input],
                outputs=output_audio
            )

        # ── Tab 2: Audio → Re-voiced Speech ─────────────────────────────────
        with gr.Tab("🔁 Audio → Clone"):
            with gr.Row():
                with gr.Column(scale=1):
                    source_audio_input = gr.Audio(
                        label="1. Source Audio (speech to re-voice)",
                        type="filepath"
                    )

                    ref_audio_convert_input = gr.Audio(
                        label="2. Reference Voice (target speaker)",
                        type="filepath"
                    )

                    convert_btn = gr.Button("🔁 Convert Voice", variant="primary")

                with gr.Column(scale=1):
                    convert_output_audio = gr.Audio(
                        label="Converted Audio",
                        interactive=False,
                        autoplay=False
                    )

                    gr.Markdown(
                        """
                        <br>

                        ### 💡 How it works:
                        * Upload any speech recording as the **source**.
                        * Upload a short clip of the **target speaker** as the reference.
                        * KokoClone re-voices the source speech to sound like the reference — no transcription needed.

                        ### Tips:
                        * Clean, noise-free audio works best for both inputs.
                        * Reference clips of 3–10 seconds give the best voice transfer.
                        """
                    )

            convert_btn.click(
                fn=convert_voice,
                inputs=[source_audio_input, ref_audio_convert_input],
                outputs=convert_output_audio
            )

# 4. Launch the app
if __name__ == "__main__":
    # Gradio 6.0 fix: Moved theme here and removed show_api
    demo.launch(server_name="0.0.0.0")
