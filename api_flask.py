from api_llm_scoring import LLMScoring
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

scorer = LLMScoring('readerbench/qwen2_1.5b_scoring_se_ta_su_pa_v3', 'cuda')

@app.route('/score/<task>', methods=['POST'] )
def score(task):
    if task not in ["selfexplanation", "thinkaloud", "summary", "paraphrasing"]:
        return "Invalid Task (should be one of: 'selfexplanation', 'thinkaloud', 'summary', 'paraphrasing')", 400
    args = request.json
    try:
        prediction = scorer.score(args, task)
    except ValueError as e:
        return str(e), 400
    response = jsonify(prediction)
    return response, 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)