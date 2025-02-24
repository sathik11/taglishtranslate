import os  
from xml.etree import ElementTree as ET
from openai import AzureOpenAI  
from azure.core.credentials import AzureKeyCredential
from azure.ai.inference.models import SystemMessage, UserMessage
import azure.cognitiveservices.speech as speechsdk
from azure.storage.blob import BlobServiceClient
import uuid
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

load_dotenv()
endpoint = os.getenv("ENDPOINT_URL", "https://aiservices-eastus-demo.services.ai.azure.com/models")  
deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4o-20nov24")  
aikey = os.getenv("AZURE_AI_ENDPOINT_KEY")

# Load environment variables from a .env file

client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(aikey),
) 

SYSTEM_PROMPT = """
## Who you are 
    You are an AI Taglish Translator who is expert in translating pure english to filipino context. You work for a Bank named "BDO" Full form of it is Banco De Oro.
     
 
## what you must do \nTranslate incoming text into taglish ie. causual filipino text. 
        Make sure you add azure text to speech SSML that best fits the state of text.
        ** You are receiving text only for translation so DO NOT add any additional sentences. 
  -  SSML markup must follow azure speech to text . voice is \"fil-PH-BlessicaNeural\" & xml:lang is fil-PH\" , 
  - Tonality must be casual and professional and must reflect bank customer service scenarios. so add SSML attributes accordingly. 
  - Always express numbers in english words. if its telephone number return it as every number as english word
    e.g (02) 6321 8000 must be translated as (zero two) six three two one eight zero zero zero.

#  What you carefully consider
   - Text should never have be in tagalog alone . Numbers should always be in english and expressed in words. 
   - Whenever your user provided text has uppercase BDO - transform it to Banco De Oro
   - Translation must always be taglish. You can change the order of sentence to make it grammatically /colloquially correct for taglish, but dont add any additional context or change the meaning of the sentence.
   - Try to add emphasis/silence/break bookmark attributes wherever appropriate. 
   - Provide only proper SSML tag result format. 
   
#### Guidance on Azure SSML as per their documentation - This is for your reference: 
    ```\nSome examples of contents that are allowed in each element are described in the following list:
        audio: The body of the audio element can contain plain text or SSML markup that's spoken if the audio file is unavailable or unplayable. The audio element can also contain text and the following elements: audio, break, p, s, phoneme, prosody, say-as, and sub.
        bookmark: This element can't contain text or any other elements.\nbreak: This element can't contain text or any other elements.
        emphasis: This element can contain text and the following elements: audio, break, emphasis, lang, phoneme, prosody, say-as, and sub.
        lang: This element can contain all other elements except mstts:backgroundaudio, voice, and speak.
        mstts:embedding: This element can contain text and the following elements: audio, break, emphasis, lang, phoneme, prosody, say-as, and sub.
        mstts:express-as: This element can contain text and the following elements: audio, break, emphasis, lang, phoneme, prosody, say-as, and sub.
        mstts:silence: This element can't contain text or any other elements.\nmstts:viseme: This element can't contain text or any other elements.
        p: This element can contain text and the following elements: audio, break, phoneme, prosody, say-as, sub, mstts:express-as, and s.
        phoneme: This element can only contain text and no other elements.
        prosody: This element can contain text and the following elements: audio, break, p, phoneme, prosody, say-as, sub, and s.
        s: This element can contain text and the following elements: audio, break, phoneme, prosody, say-as, mstts:express-as, and sub.
        say-as: This element can only contain text and no other elements.\nsub: This element can only contain text and no other elements.
        speak: The root element of an SSML document. This element can contain the following elements: mstts:backgroundaudio and voice.
        voice: This element can contain all other elements except mstts:backgroundaudio and speak.
        strength: The relative duration of a pause by using one of the following values:\nx-weak, weak, medium (default), strong, x-strong
        time: \tThe absolute duration of a pause in seconds (such as 2s) or milliseconds (such as 500ms). Valid values range from 0 to 20000 milliseconds. If you set a value greater than the supported maximum, the service uses 20000ms. If the time attribute is set, the strength attribute is ignored.
        About silence : Use the mstts:silence element to insert pauses before or after text, or between two adjacent sentences.\ntype\tSpecifies where and how to add silence. 
            The following silence types are supported:\nLeading – Extra silence at the beginning of the text. The value that you set is added to the natural silence before the start of text. 
            Leading-exact – Silence at the beginning of the text. The value is an absolute silence length.
            Tailing – Extra silence at the end of text. The value that you set is added to the natural silence after the last word.\nTailing-exact – Silence at the end of the text. The value is an absolute silence length.
            Sentenceboundary – Extra silence between adjacent sentences. The actual silence length for this type includes the natural silence after the last word in the previous sentence, the value you set for this type, and the natural silence before the starting word in the next sentence.\nSentenceboundary-exact – Silence between adjacent sentences. The value is an absolute silence length.\nComma-exact – Silence at the comma in half-width or full-width format. The value is an absolute silence length.
            Semicolon-exact – Silence at the semicolon in half-width or full-width format. The value is an absolute silence length.
            Enumerationcomma-exact – Silence at the enumeration comma in full-width format. The value is an absolute silence length.\n\nAn absolute silence type (with the -exact suffix) replaces any otherwise natural leading or trailing silence.
            Absolute silence types take precedence over the corresponding non-absolute type. For example, if you set both Leading and Leading-exact types, the Leading-exact type takes effect.
            The WordBoundary event takes precedence over punctuation-related silence settings including Comma-exact, Semicolon-exact, or Enumerationcomma-exact. When you use both the WordBoundary event and punctuation-related silence settings, the punctuation-related silence settings don't take effect.
"""

USER_MSG = """The available information does not mention an option 5 for Fee Waive. However, you can contact the BDO Customer Contact Center at (02) 631-8000 for further assistance with fee waivers and other inquiries."""

def process_voice_tag(xml_string):
    # Remove markdown ``` and ssml if present
    xml_string = xml_string.replace("`ssml", "").replace("`xml", "").replace("`", "").strip()
    try:
        root = ET.fromstring(xml_string)
    except ET.ParseError as e:
        print(f"Error parsing XML: {e}")
        return None, None
    
    # Parse the XML
    namespace = {"ssml": "http://www.w3.org/2001/10/synthesis"}  # Define the namespace
    root = ET.fromstring(xml_string)

    # Extract contents inside <voice> tag
    voice_element = root.find("ssml:voice", namespace)  # Find the <voice> tag
    voice_tag = ET.tostring(voice_element, encoding="unicode", method="xml")
    # Step 1: Attach voice tag to speech tag
    pretag = """<speak xmlns="http://www.w3.org/2001/10/synthesis" version="1.0" xml:lang="fil-PH">"""
    posttag = """</speak>"""
    speech_tag_string = f"{pretag}{voice_tag}{posttag}"

    # Step 2: Extract text and remove inner tags
    text_content = ''.join(voice_element.itertext())

    #remove extra whitespace
    text_content = " ".join(text_content.split())

    return speech_tag_string, text_content

def generate_audio_store(speechinput,str_prefix):
        
    speech_config = speechsdk.SpeechConfig(subscription=os.getenv("AZURE_SPEECH_KEY"), region=os.getenv("AZURE_SPEECH_REGION"))

    # use the default speaker as audio output.
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)

    speech_synthesis_result = speech_synthesizer.speak_ssml_async(speechinput).get()
    stream = speechsdk.AudioDataStream(speech_synthesis_result)
    # Save the audio stream to a local file
    local_file_name = f"{str_prefix}-{uuid.uuid4()}.wav"
    stream.save_to_wav_file(local_file_name)

    # Upload the file to Azure Blob Storage
    blob_service_client = BlobServiceClient.from_connection_string(os.getenv("AZURE_STORAGE_CONNECTION_STRING"))
    container_name = os.getenv("AZURE_STORAGE_CONTAINER_NAME")
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=local_file_name)

    with open(local_file_name, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)

    sas_token = os.getenv("AZURE_STORAGE_SAS_TOKEN")
    blob_url_with_sas = f"{blob_client.url}?{sas_token}"
    print(f"Blob URL with SAS token: {blob_url_with_sas}")
    # Delete the local file after upload
    os.remove(local_file_name)
    return blob_url_with_sas


def taglish_translate(usermsg):


    response = client.complete(
        messages=[
            SystemMessage(content=SYSTEM_PROMPT),
            UserMessage(content=usermsg),
        ],
        model=deployment
    )

    # print(response.choices[0].message.content)

    ai_output = response.choices[0].message.content
    # print(ai_output)
    speech_output, text_output = process_voice_tag(ai_output)
    blob_url_with_sas = generate_audio_store(speech_output,text_output[:3])

    return {"speech_output":blob_url_with_sas, "text_output":text_output}