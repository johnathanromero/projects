from flask import Flask, request, render_template, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
import openai
from google.cloud import storage
import os
import re
from io import BytesIO
from io import BufferedIOBase
import numpy as np
import ffmpeg
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from werkzeug.utils import secure_filename
import tempfile
from moviepy.editor import VideoFileClip
import time
from pydub import AudioSegment
from google.cloud import translate_v2 as translate
import logging

class NamedBytesIO(BytesIO, BufferedIOBase):
    def __init__(self, data, name=None):
        super().__init__(data)
        self.name = name

app = Flask(__name__)

logger = logging.getLogger(__name__)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ""
storage_client = storage.Client()
BUCKET_NAME = "xxxx"
bucket = storage_client.get_bucket(BUCKET_NAME)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_upload_url', methods=['GET'])
def generate_upload_url():
    filename = request.args.get('filename')
    content_type = request.args.get('contentType')

    if not filename or not content_type:
        return jsonify({'error': 'filename and contentType are required'}), 400

    blob = bucket.blob(filename)
    url = blob.generate_signed_url(
        version="v4",
        expiration=3600,  # Signed URL will be valid for 1 hour
        method="PUT",
        content_type=content_type,
    )

    return jsonify({"url": url})



@app.route('/transcribe_summarize/<file_name>')
def transcribe_summarize(file_name):
    return render_template('transcribe_summarize.html', file_name=file_name)





def google_translate_text(text, target_language):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "xxx"
    translate_client = translate.Client()
    result = translate_client.translate(text, target_language=target_language)
    translation = result['translatedText']
    return translation



@app.route('/process_file', methods=['GET'])
def process_file():
    if request.method == 'GET':
        language = request.args.get('language')        
        file_name = request.args.get('filename')
        if file_name is None:
            return "No filename provided", 400    
        BUCKET_NAME = "xxxx"
        
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS
        storage_client = storage.Client()

        bucket = storage_client.get_bucket(BUCKET_NAME)
        blob = bucket.blob(file_name)
        local_file_path = f'/tmp/{file_name}'

        blob.download_to_filename(local_file_path)

        CHUNK_SIZE = 1024 * 1024 * 20  # 20 MB
        transcript_chunks = []
        summary_chunks = []
        
        if os.path.splitext(local_file_path)[1] == '.mp4':
            output_file_path = local_file_path[:-3] + '_converted.mp3'
            ffmpeg.input(local_file_path).output(output_file_path, y='-y').run()
        else:
            output_file_path = local_file_path

        try:
            with open(output_file_path, 'rb') as audio_file:
                while True:
                    chunk = audio_file.read(CHUNK_SIZE)
                    if not chunk:
                        break

                    media_bytes = NamedBytesIO(chunk, name='chunk.wav')
                    response = openai.Audio.transcribe(
                        api_key=os.environ.get('WHISPER_API_KEY'),
                        model='whisper-1',
                        file=media_bytes,
                    )
                    ##get transcribed text chunk
                    transcribed_chunk = response['text']

                    ##test for language
                    if language.lower() != 'en':
                        translated_text = google_translate_text(transcribed_chunk, language)
                        transcript_chunks.append(translated_text)                    
                    else:
                        transcript_chunks.append(transcribed_chunk)

                    
                    # transcript_chunks.append(transcribed_chunk)
                    
                    openai.api_key = os.environ.get('OPEN_API_KEY')                
                    
                    # Generate summary for the transcribed_chunk
                    prompt = f"""Analyze the transcript and provide the following:
                    - Main talking points (max 5, 50 words each)
                    - Main follow up items (max 5, 50 words each)
                    - Main un-answered questions (max 5, 50 words each)
                    - Sentiment analysis
                    Ensure the final element of any array is not followed by a comma:

                    {transcribed_chunk}
                    """

                    response = openai.ChatCompletion.create(
                        model="gpt-4-32k",
                        messages = [{"role": "system", "content": prompt}])
                    
                    summary_chunk = response['choices'][0]['message']['content']
                    summary_chunks.append(summary_chunk)
                    
            
            transcript = ''.join(transcript_chunks)
            


            # list of OpenAI/ChatGPT responses
            responses = summary_chunks

            # create CountVectorizer object
            vectorizer = CountVectorizer()

            # convert responses to a document-term matrix
            response_matrix = vectorizer.fit_transform(responses)

            # calculate cosine similarity between each response
            similarity_matrix = cosine_similarity(response_matrix)

            # get the index of the most similar response
            most_similar_index = np.argmax(similarity_matrix.mean(axis=0))

            # select the most similar response as the final response
            final_response = responses[most_similar_index]

            #print(final_response)

            # Add new lines before any number or colon
            # Add a new line after any colon.
            summary_text = re.sub(r':(\s*|$)', r'\n\1', final_response)


            transcript = re.findall(r"[^.!?]+[.!?]+", transcript)
            transcript = "\n\n".join(transcript)

            blob = bucket.blob(file_name)
            if blob.exists():
                logger.info(f"Deleting blob: {blob.name}")
                blob.delete()
            else:
                logger.error(f"Blob not found: {blob.name}")


            return render_template('job_complete.html', summary_text=summary_text, transcript=transcript)
       
        except Exception as e:
            logger.error(f"Error in function: {e}")
            # Attempt to delete the blob in case of an exception
            blob = bucket.blob(file_name)
            if blob.exists():
                logger.info(f"Deleting blob due to error: {blob.name}")
                blob.delete()
            else:
                logger.error(f"Blob not found during error handling: {blob.name}")

            raise
        finally:
            # Delete the local files
            if os.path.exists(output_file_path):
                logger.info(f"Deleting local file: {output_file_path}")
                os.remove(output_file_path)
            else:
                logger.error(f"Local file not found: {output_file_path}")

            if os.path.exists(local_file_path):
                logger.info(f"Deleting local file: {local_file_path}")
                os.remove(local_file_path)
            else:
                logger.error(f"Local file not found: {local_file_path}")        
    else:
        return render_template('process_file.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
