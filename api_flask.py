import io
import xlsxwriter
from learning_strategies_scoring.api_llm_scoring import LLMScoring
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

from quiz_generation.quiz_generation import QuizGeneration

app = Flask(__name__)
CORS(app)
device = "cuda"
scorer = LLMScoring('upb-nlp/llama32_3b_scoring_all_tasks', device=device)
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

def export_questions(context, questions):
    buffer = io.BytesIO()
    with xlsxwriter.Workbook(buffer) as workbook:
        worksheet = workbook.add_worksheet()
        worksheet.write_row(0, 0, ["Context", "Question", "Answer", "Distractor 1", "Distractor 2", "Distractor 3"])
        for i, question in enumerate(questions):
            worksheet.write_string(i+1, 0, context)
            worksheet.write_string(i+1, 1, question["question"])
            worksheet.write_string(i+1, 2, question["answer"])
            worksheet.write_string(i+1, 3, question["distractors"][0])
            worksheet.write_string(i+1, 4, question["distractors"][1])
            worksheet.write_string(i+1, 5, question["distractors"][2])
            
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name="questions.xlsx")

@app.route('/quiz/generate', methods=['POST'] )
def generate_quiz():
    args = request.json
    if "context" not in args:
        return "Context not provided", 400
    context = args["context"]
    if len(context.split()) > 1000:
        return "Maximum context length = 1000 words", 400
    model = args.get("model", "3B")
    qg = quiz_generators[model]
    num_questions = args.get("num_questions", 10)
    strategy = args.get("strategy", "all")
    if strategy == "pipeline":
        num_samples = args.get("num_samples", 10)
        questions = qg.generate_quiz_pipeline(context=context, num_final_questions=num_questions, num_samples_answers=num_samples, num_samples_distractors=num_samples)
    else:
        num_samples = args.get("num_samples", 40)
        questions = qg.generate_quiz_everything(context=context, num_samples=num_samples, num_final_questions=num_questions)
    questions = [
        {
            "question": question["question"],
            "answer": question["answer"],
            "distractors": question["distractors"],
        }
        for question in questions
    ]
    if args.get("export", False):
        return export_questions(context, questions), 200
    else:
        return jsonify(questions), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)