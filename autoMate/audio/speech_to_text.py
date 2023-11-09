
from io import BytesIO
from os import system, name
from speech_recognition import Recognizer, Microphone, AudioData
from whisper import load_model
from torch.cuda import is_available

from datetime import datetime, timedelta
from queue import Queue
from tempfile import NamedTemporaryFile
from time import sleep
from autoMate.core.config import ConfigDict

MODEL_CONFIG = 'Model'
ENERGYTHRSHLD_CONFIG = 'EnergyThreshold'
RECORDTIMEOUT_CONFIG = 'RecordTimeout'
PHRASETIMEOUT_CONFIG = 'PhraseTimeout'
TRANSCRIPTLOG_CONFIG = 'AudioTranscript'

class SpeechToText():
    def __init__(self, config : ConfigDict):
        self.config = config
        print("DOWNLOAD")
        self.audio_model = self._download_model(self.config[MODEL_CONFIG])
        print("END")

    def _download_model(self, model : str):
        # TODO add enum to represent model instead of str
        model = self.config[MODEL_CONFIG]
        if model != "large":
            model = model + ".en"
        return load_model(model)
    
    def run_model(self):            
        # The last time a recording was retrieved from the queue.
        phrase_time = None
        # Current raw audio bytes.
        last_sample = bytes()
        # Thread safe Queue for passing data from the threaded recording callback.
        data_queue = Queue()
        # We use SpeechRecognizer to record our audio because it has a nice feature where it can detect when speech ends.
        recorder = Recognizer()
        recorder.energy_threshold = self.config[ENERGYTHRSHLD_CONFIG]
        # Definitely do this, dynamic energy compensation lowers the energy threshold dramatically to a point where the SpeechRecognizer never stops recording.
        recorder.dynamic_energy_threshold = True

        source = Microphone(sample_rate=16000)

        # Load / Download model
        model = self.config[MODEL_CONFIG]
        if model != "large":
            model = model + ".en"
        audio_model = load_model(model)

        record_timeout = self.config[RECORDTIMEOUT_CONFIG]
        phrase_timeout = self.config[PHRASETIMEOUT_CONFIG]

        temp_file = NamedTemporaryFile().name
        transcription = ['']

        with source:
            recorder.adjust_for_ambient_noise(source)

        def record_callback(_, audio:AudioData) -> None:
            """
            Threaded callback function to receive audio data when recordings finish.
            audio: An AudioData containing the recorded bytes.
            """
            # Grab the raw bytes and push it into the thread safe queue.
            data = audio.get_raw_data()
            data_queue.put(data)

        # Create a background thread that will pass us raw audio bytes.
        # We could do this manually but SpeechRecognizer provides a nice helper.
        recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

        # Cue the user that we're ready to go.
        print("Model loaded.\n")

        # TODO run this in own thread and pass transcription through asyncio
        while True:
            try:
                now = datetime.utcnow()
                # Pull raw recorded audio from the queue.
                if not data_queue.empty():
                    phrase_complete = False
                    # If enough time has passed between recordings, consider the phrase complete.
                    # Clear the current working audio buffer to start over with the new data.
                    if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                        last_sample = bytes()
                        phrase_complete = True
                    # This is the last time we received new audio data from the queue.
                    phrase_time = now

                    # Concatenate our current audio data with the latest audio data.
                    while not data_queue.empty():
                        data = data_queue.get()
                        last_sample += data

                    # Use AudioData to convert the raw data to wav data.
                    audio_data = AudioData(last_sample, source.SAMPLE_RATE, source.SAMPLE_WIDTH)
                    wav_data = BytesIO(audio_data.get_wav_data())

                    # Write wav data to the temporary file as bytes.
                    with open(temp_file, 'w+b') as f:
                        f.write(wav_data.read())

                    # Read the transcription.
                    result = audio_model.transcribe(temp_file, fp16=is_available())
                    text = result['text'].strip()

                    # If we detected a pause between recordings, add a new item to our transcription.
                    # Otherwise edit the existing one.
                    if phrase_complete:
                        transcription.append(text)
                    else:
                        transcription[-1] = text

                    # Clear the console to reprint the updated transcription.
                    system('cls' if name=='nt' else 'clear')
                    for line in transcription:
                        print(line)
                    # Flush stdout.
                    print('', end='', flush=True)

                    # Infinite loops are bad for processors, must sleep.
                    sleep(0.25)
            except KeyboardInterrupt:
                break

        # At end of program, log transcript
        # TODO async write to file and clear old data
        with open(self.config[TRANSCRIPTLOG_CONFIG], "w") as transcript:
            transcript.write("Transcription [%s]:\n" % datetime.utcnow())
            for line in transcription:
                transcript.write("%s\n" % line)