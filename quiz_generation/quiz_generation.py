from question_generation_utils import QuestionGenerationUtils

class QuizGeneration:
    def __init__(self, device, model_name):
        if device not in ['mps', 'cuda', 'cpu']:
            raise ValueError('Invalid device. Choose from: mps, cuda, cpu')
        if not model_name:
            raise ValueError('model_name cannot be empty')
        
        self.qg_utils = QuestionGenerationUtils(device, model_name)

    def _levenstein_distance(self, str1, str2):
        m = len(str1)
        n = len(str2)

        dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]

        for i in range(m + 1):
            for j in range(n + 1):
                if i == 0:
                    dp[i][j] = j
                elif j == 0:
                    dp[i][j] = i
                elif str1[i - 1] == str2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

        return dp[m][n]

    def _is_duplicate(self, str1, str2):
        if self._levenstein_distance(str1.lower(), str2.lower()) <= 1:
            return True
        return False
    
    def _eliminate_duplicates(self, answers, losses):
        answers_dedup = []
        losses_dedup = []

        for i in range(len(answers)):
            duplicate = False
            for j in range(len(answers_dedup)):
                if self._is_duplicate(answers[i], answers_dedup[j]):
                    duplicate = True
                    break
            if not duplicate:
                answers_dedup.append(answers[i])
                losses_dedup.append(losses[i])

        return answers_dedup, losses_dedup

    def generate_quiz_pipeline(self, context, num_samples_questions=20, num_samples_answers=10, num_chosen_answers=2, num_samples_distractors=10, num_final_questions=10):
        response = []
        questions, qgen_lossess = self.qg_utils.generate_all_questions(context, num_samples_questions)

        questions = [q.strip() for q in questions]

        for question, qgen_loss in zip(questions, qgen_lossess):
            try:
                answers, qa_lossess = self.qg_utils.generate_all_answers(context, question, num_samples_answers)

                # Filter out answers and losses with qa_loss > 5
                answers = [a for a, loss in zip(answers, qa_lossess) if loss <= 5]
                qa_lossess = [loss for loss in qa_lossess if loss <= 5]

                # Sort answers by qa_loss
                answers = [a.strip() for _, a in sorted(zip(qa_lossess, answers), key=lambda pair: pair[0])]
                qa_lossess = [loss for loss in sorted(qa_lossess)]

                answers, qa_lossess = self._eliminate_duplicates(answers, qa_lossess)

                answers = answers[:num_chosen_answers]

                answers_distractors_candidates = []
                for answer, qa_loss in zip(answers, qa_lossess):
                    distractors, dgen_lossess = self.qg_utils.generate_all_distractors(context, question, answer, num_samples_distractors)

                    # Sort distractors and losses by the max(dgen_loss)
                    distractors = [d for _, d in sorted(zip(dgen_lossess, distractors), key=lambda pair: max(pair[0]))]
                    for i in range(len(distractors)):
                        distractors[i] = [d.strip() for d in distractors[i]]
                    dgen_lossess = [loss for loss in sorted(dgen_lossess, key=lambda x: max(x))]

                    for distractor_set, dgen_loss_set in zip(distractors, dgen_lossess):
                        qa_loss_distactors = self.qg_utils.get_qa_loss(context, question, distractor_set)
                        # If there is every element from qa_loss_distactors is greater than qa_loss
                        if all([qa_loss_distactor - qa_loss > 0.5 for qa_loss_distactor in qa_loss_distactors]):
                            answers_distractors_candidates.append({
                                'answer': answer,
                                'qa_loss': qa_loss,
                                'distractors': distractor_set,
                                'dgen_loss': dgen_loss_set,
                                'qa_loss_distractors': qa_loss_distactors,
                            })
                            break

                # Sort answers_distractors_candidates by max(dgen_loss)
                answers_distractors_candidates_sorted = sorted(answers_distractors_candidates, key=lambda x: max(x['dgen_loss']) + x['qa_loss'])

                chosen_answer_distractors = answers_distractors_candidates_sorted[0]

                response.append({
                    'question': question,
                    'answer': chosen_answer_distractors['answer'],
                    'distractors': chosen_answer_distractors['distractors'],
                    'qgen_loss': qgen_loss,
                    'qa_loss': chosen_answer_distractors['qa_loss'],
                    'dgen_loss': chosen_answer_distractors['dgen_loss'],
                    'qa_loss_distractors': chosen_answer_distractors['qa_loss_distractors'],
                })

            except:
                pass

        response = sorted(response, key=lambda x: x['qgen_loss'] + 2 * x['qa_loss'] - min(x['qa_loss_distractors']))
        return response[:num_final_questions]