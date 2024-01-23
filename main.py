import os
from flask import Flask, request
from flask_cors import CORS
import whisper

app = Flask(__name__)
CORS(app)

# temporary folder to store uploaded file
UPLOAD_FOLDER = 'C:/temp_whisper_uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/', methods=['GET'])
def hello_world():
    return 'Hello World!'


@app.route('/stt', methods=['POST'])
def speech_to_text_api():
    try:
        if 'speech' not in request.files:
            return 'No file part'

        audio_file = request.files['speech']
        if audio_file.filename == '':
            return 'No selected file'

        # Save the uploaded file with a unique filename
        unique_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_audio.wav')
        audio_file.save(unique_filename)

        model = whisper.load_model("base")

        # load audio and pad/trim it to fit 30 seconds
        audio = whisper.load_audio(unique_filename)
        audio = whisper.pad_or_trim(audio)

        # make log-Mel spectrogram and move to the same device as the model
        mel = whisper.log_mel_spectrogram(audio).to(model.device)

        # detect the spoken language
        _, probs = model.detect_language(mel)
        print(f"Detected language: {max(probs, key=probs.get)}")

        # decode the audio
        options = whisper.DecodingOptions(fp16 = False)
        result = whisper.decode(model, mel, options)

        # print the recognized text
        print(result.text)
        return result.text

    except Exception as e:
        return f'Error: {e}'


if __name__ == '__main__':
    app.run(debug=True)