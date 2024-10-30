from learning_strategies_scoring.api_llm_scoring import LLMScoring
from flask import Flask, request, jsonify
from flask_cors import CORS

from quiz_generation.quiz_generation import QuizGeneration

app = Flask(__name__)
CORS(app)
device = "cuda"
scorer = LLMScoring('readerbench/qwen2_1.5b_scoring_se_ta_su_pa_v3', device=device)
quiz_generators = {
    "1B": QuizGeneration(device=device, model_name="readerbench/llama3.2_1b_instruct_qall_lr_small"),
    "3B": QuizGeneration(device=device, model_name="readerbench/llama3.2_3b_instruct_qall_lr_small"),
}

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

@app.route('/quiz/generate', methods=['POST'] )
def generate_quiz():
    args = request.json
    if "context" not in args:
        return "Context not provided", 400
    context = args["context"]
    model = args.get("model", "3B")
    qg = quiz_generators[model]
    num_questions = args.get("num_questions", 10)
    questions = qg.generate_quiz_pipeline(context=context, num_final_questions=num_questions)
    questions = [
        {
            "question": question["question"],
            "answer": question["answer"],
            "distractors": question["distractors"],
        }
        for question in questions
    ]
    response = jsonify(questions)
    return response, 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)