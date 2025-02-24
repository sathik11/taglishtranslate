from flask import Flask, request, jsonify
import logging
from translate_ssml import taglish_translate
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/taglishtranslator', methods=['POST'])
def taglishtranslator():
    logging.info('Flask HTTP POST request received.')

    try:
        req_body = request.get_json()
        eng_text = req_body['englishtext']
        taglish_res = taglish_translate(eng_text)
    except (ValueError, KeyError) as e:
        logging.error(f"Error processing request: {e}")
        return jsonify({"error": "Invalid input"}), 400

    if taglish_res:
        return jsonify(taglish_res), 200
    else:
        return jsonify({
            "message": "This HTTP triggered function executed successfully. Pass a name in the query string or in the request body for a personalized response."
        }), 200

if __name__ == '__main__':
    app.run(port=5000)