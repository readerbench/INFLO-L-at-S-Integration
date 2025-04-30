from api_llm_scoring import LLMScoring

llm_scoring = LLMScoring('upb-nlp/llama32_3b_scoring_all_tasks', 'cuda')

my_data = {
    'target_sentence': "Sometimes blood does not transport enough oxygen, resulting in a condition called anemia.",
    'context': "Red blood cells have the vital role of carrying oxygen to all of the cells in the body.  They also pick up waste carbon dioxide for removal.  These cells are the most numerous of the blood cells.  The disk shape of red blood cells results in a large surface area, which enables them to be efficient at gas diffusion.\nRed blood cells contain a large, complex protein called hemoglobin.  Hemoglobin binds to the oxygen and carbon dioxide that the red blood cells transport.  Each red blood cell contains about 250 million hemoglobin molecules, each carrying four molecules of oxygen.  Hemoglobin also contains iron, which gives blood its red color.  Molecular oxygen can also be transported by another route, dissolved in blood plasma. However, oxygen is poorly soluble in water, so only about 1.5% is carried in dissolved form.  Therefore, most oxygen is carried by hemoglobin.\nRed blood cells lack a nucleus and the organelles found in other cells.  Therefore, these cells cannot reproduce or repair themselves.  Red blood cells live for about three or four months before being broken down in the spleen.  Iron from the broken-down cells is returned to the bone marrow to be recycled into new hemoglobin.\nSometimes blood does not transport enough oxygen, resulting in a condition called anemia. This makes a person feel tired and weak.  Anemia can result from too little iron in the diet, loss of blood due to injury or menstruation, or various medical conditions.  One type of anemia, called sickle-cell disease, is characterized by red blood cells that are sickle-shaped instead of disk-shaped.  The shape of the cells causes them to clog blood vessels, preventing oxygen from reaching muscles and other tissues.",
    'student_response': "You develop a condition because you did not have enough oxygen transported, which could mean your red blood cells aren't functioning right.",
}

task = 'selfexplanation'

scores = llm_scoring.score(my_data, task)
print(scores)
