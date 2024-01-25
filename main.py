import os
from dotenv import load_dotenv
from flask import Flask, request
from flask_cors import CORS
import whisper
from openai import OpenAI
import httpx
from pathlib import Path


app = Flask(__name__)
CORS(app)
load_dotenv()

# chat-gpt
client = OpenAI(api_key=os.getenv("CHAT_GPT_KEY"))

# temporary folder to store uploaded file
UPLOAD_FOLDER = 'C:/temp_whisper_uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

DOWNLOAD_FOLDER = 'C:/temp_chat_gpt_downloads'
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER


async def send_message(prompt):

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        model="gpt-3.5-turbo"
    )

    reply = chat_completion.choices[0].message.content
    print(reply)

    speech_file_path = Path(__file__).parent / "speech.mp3"
    result = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=reply
    )
    result.stream_to_file(speech_file_path)


    await text_to_speech(reply)


async def text_to_speech(text):
    url = "https://api.openai.com/v1/audio/speech"
    headers = {
        "Authorization": "Bearer " + os.getenv("CHAT_GPT_KEY")
    }
    data = {
        "model": "tts-1",
        "voice": "alloy",
        "input": text,
        "max_tokens": 150
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=data)

        if response.status_code == 200:
            # Assuming the API response contains the speech data
            speech_data = response.content

            reply_filename = os.path.join(app.config['DOWNLOAD_FOLDER'], 'downloaded_audio.wav')

            os.makedirs(os.path.dirname(reply_filename), exist_ok=True)
            with open(reply_filename, 'wb') as audio_file:
                audio_file.write(speech_data)

            return reply_filename  # Return the file path or any other indicator of success
        else:
            print(f"Error: {response.status_code}, {response.text}")
            return None


@app.route('/', methods=['GET'])
def hello_world():
    return 'Hello World!'


@app.route('/stt', methods=['POST'])
async def speech_to_text_api():
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
        print(f"filename: {audio_file.filename}")
        print(f"Detected language: {max(probs, key=probs.get)}")

        # decode the audio
        options = whisper.DecodingOptions(fp16 = False)
        result = whisper.decode(model, mel, options)

        # print the recognized text
        print(result.text)

        # get answer from chat-GPT
        await send_message(result.text)
        return result.text

    except Exception as e:
        return f'Error: {e}'


if __name__ == '__main__':
    app.run(debug=True)