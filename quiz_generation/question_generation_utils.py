from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class QuestionGenerationUtils:
    def __init__(self, device, model_name):
        if device not in ['cpu', 'cuda', 'mps']:
            raise ValueError(f"Invalid device: {device}. Must be one of 'cpu', 'cuda', 'mps'.")
        if not model_name:
            raise ValueError("Model name must be provided.")
        
        self.device = device
        self.qall_tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.qall_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        self.qall_tokenizer.pad_token_id = self.qall_tokenizer.eos_token_id
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    def _find_sublist_index(self, main_list, sublist):
        n, m = len(main_list), len(sublist)
        for i in range(n - m + 1):
            if main_list[i:i + m] == sublist:
                return i
        return -1
    
    def _cut_sublist(self, main_list, num_elements, end_idx):
        sublist = main_list[:end_idx]
        res = sublist[-num_elements:]
        return res
    
    def generate_all_questions(self, context, num_samples):
        messages = [
            {"role": "system", "content": "You are an educational expert."},
            {"role": "user", "content": f"Generate a question, an answer and 3 distractors based on the context.\n\nContext:\n{context}"},
            {"role": "assistant", "content": f"Question:"},
        ]
        prompt = self.qall_tokenizer.apply_chat_template(messages, add_special_tokens=False, tokenize=False, continue_final_message=True)

        inputs = self.qall_tokenizer([prompt], return_tensors="pt", max_length=2048, truncation=True, padding='longest').to(self.device)

        with torch.no_grad():
            outputs = self.qall_model.generate(
                inputs['input_ids'], 
                attention_mask=inputs['attention_mask'], 
                do_sample=True,
                temperature=None,
                top_k=None, 
                top_p=None,
                max_new_tokens=40, 
                num_return_sequences=num_samples,
                pad_token_id=self.qall_tokenizer.eos_token_id,
                tokenizer=self.qall_tokenizer,
                stop_strings=["Answer:"],
                output_logits=True,
                return_dict_in_generate=True,
            )
            stop_string_ids = self.qall_tokenizer.encode("Answer:", add_special_tokens=False)

            generated_ids_sequences = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip([inputs.input_ids[0]] * num_samples, outputs['sequences'])
            ]

            generated_ids_logits = torch.transpose(torch.stack(outputs['logits']), 0, 1)

            losses = []

            for i in range(len(generated_ids_sequences)):
                end_idx_sequence = self._find_sublist_index(generated_ids_sequences[i].tolist(), stop_string_ids)
                generated_ids_sequences[i] = generated_ids_sequences[i][:end_idx_sequence]

                good_logit = generated_ids_logits[i][:end_idx_sequence]

                loss = self.loss_fn(
                        good_logit,
                        generated_ids_sequences[i],
                )
                losses.append(loss.item())

            responses = self.qall_tokenizer.batch_decode(generated_ids_sequences, skip_special_tokens=True)

        return responses, losses
    
    def generate_all_answers(self, context, question, num_samples):
        messages = [
            {"role": "system", "content": "You are an educational expert."},
            {"role": "user", "content": f"Generate a question, an answer and 3 distractors based on the context.\n\nContext:\n{context}"},
            {"role": "assistant", "content": f"Question: {question}\n\nAnswer:"},
        ]
        prompt = self.qall_tokenizer.apply_chat_template(messages, add_special_tokens=False, tokenize=False, continue_final_message=True)

        inputs = self.qall_tokenizer([prompt], return_tensors="pt", max_length=2048, truncation=True, padding='longest').to(self.device)

        with torch.no_grad():
            outputs = self.qall_model.generate(
                inputs['input_ids'], 
                attention_mask=inputs['attention_mask'], 
                do_sample=True,
                temperature=None,
                top_k=None, 
                top_p=None,
                max_new_tokens=40, 
                num_return_sequences=num_samples,
                pad_token_id=self.qall_tokenizer.eos_token_id,
                tokenizer=self.qall_tokenizer,
                stop_strings=["Distractors:\n"],
                output_logits=True,
                return_dict_in_generate=True,
            )
            stop_string_ids = self.qall_tokenizer.encode("Distractors:\n", add_special_tokens=False)

            generated_ids_sequences = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip([inputs.input_ids[0]] * num_samples, outputs['sequences'])
            ]

            generated_ids_logits = torch.transpose(torch.stack(outputs['logits']), 0, 1)

            losses = []

            for i in range(len(generated_ids_sequences)):
                end_idx_sequence = self._find_sublist_index(generated_ids_sequences[i].tolist(), stop_string_ids)
                generated_ids_sequences[i] = generated_ids_sequences[i][:end_idx_sequence]

                good_logit = generated_ids_logits[i][:end_idx_sequence]

                loss = self.loss_fn(
                        good_logit,
                        generated_ids_sequences[i],
                )
                losses.append(loss.item())

            responses = self.qall_tokenizer.batch_decode(generated_ids_sequences, skip_special_tokens=True)

        return responses, losses

    def generate_all_distractors(self, context, question, answer, num_samples):
        messages = [
            {"role": "system", "content": "You are an educational expert."},
            {"role": "user", "content": f"Generate a question, an answer and 3 distractors based on the context.\n\nContext:\n{context}"},
            {"role": "assistant", "content": f"Question: {question}\n\nAnswer: {answer}\n\nDistractors:"},
        ]
        prompt = self.qall_tokenizer.apply_chat_template(messages, add_special_tokens=False, tokenize=False, continue_final_message=True) + "\n"

        inputs = self.qall_tokenizer([prompt], return_tensors="pt", max_length=2048, truncation=True, padding='longest').to(self.device)

        with torch.no_grad():
            outputs = self.qall_model.generate(
                inputs['input_ids'], 
                attention_mask=inputs['attention_mask'], 
                do_sample=True,
                temperature=None,
                top_k=None, 
                top_p=None,
                max_new_tokens=40, 
                num_return_sequences=num_samples,
                pad_token_id=self.qall_tokenizer.eos_token_id,
                output_logits=True,
                return_dict_in_generate=True,
            )

            generated_ids_sequences = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip([inputs.input_ids[0]] * num_samples, outputs['sequences'])
            ]

            generated_ids_logits = torch.transpose(torch.stack(outputs['logits']), 0, 1)

            all_losses = []
            all_responses = []

            token_ids_with_newline = {token_id for token, token_id in self.qall_tokenizer.vocab.items() if 'Ċ' in token}

            for i in range(len(generated_ids_sequences)):
                losses = []
                distractors_ids = []
                end_idx_sequence = generated_ids_sequences[i].tolist().index(self.qall_tokenizer.eos_token_id) + 1 if self.qall_tokenizer.eos_token_id in generated_ids_sequences[i] else len(generated_ids_sequences[i])
                generated_ids_sequences[i] = generated_ids_sequences[i][:end_idx_sequence]

                good_logit = generated_ids_logits[i][:end_idx_sequence]

                indices = [i+1 for i, x in enumerate(generated_ids_sequences[i]) if x.item() in token_ids_with_newline]
                generated_ids_splitted = [generated_ids_sequences[i][j:k] for j, k in zip([0] + indices, indices + [len(generated_ids_sequences[i])])]
                generated_logits_splitted = [good_logit[j:k] for j, k in zip([0] + indices, indices + [len(good_logit)])]

                for gen_id, gen_logit in zip(generated_ids_splitted, generated_logits_splitted):
                    loss = self.loss_fn(
                        gen_logit,
                        gen_id,
                    )
                    losses.append(loss.item())
                    distractors_ids.append(gen_id)

                response = self.qall_tokenizer.batch_decode(distractors_ids, skip_special_tokens=True)

                all_losses.append(losses)
                all_responses.append(response)

        return all_responses, all_losses
    
    def get_qa_loss(self, context, question, answers):
        losses = []

        messages_input = [
            [
                {"role": "system", "content": "You are an educational expert."},
                {"role": "user", "content": f"Generate a question, an answer and 3 distractors based on the context.\n\nContext:\n{context}"},
                {"role": "assistant", "content": f"Question: {question}\n\nAnswer:"},
            ] for _ in answers
        ]
        texts_input = [self.qall_tokenizer.apply_chat_template(messages, add_special_tokens=False, tokenize=False, continue_final_message=True) for messages in messages_input]

        messages_whole = [
            [
                {"role": "system", "content": "You are an educational expert."},
                {"role": "user", "content": f"Generate a question, an answer and 3 distractors based on the context.\n\nContext:\n{context}"},
                {"role": "assistant", "content": f"Question: {question}\n\nAnswer: {ans}\n\nDistractors:"},
            ] for ans in answers
        ]
        texts_whole = [self.qall_tokenizer.apply_chat_template(messages, add_special_tokens=False, tokenize=False, continue_final_message=True) + '\n' for messages in messages_whole]

        input_prompts = self.qall_tokenizer(texts_input, return_tensors="pt", truncation=True, max_length=2048, padding='longest').to(self.device)
        whole_prompts = self.qall_tokenizer(texts_whole, return_tensors="pt", truncation=True, max_length=2048, padding='longest').to(self.device)

        with torch.no_grad():
            outputs = self.qall_model(input_ids=whole_prompts['input_ids'], attention_mask=whole_prompts['attention_mask'])
            logits = outputs.logits

        for logit, input, whole in zip(logits, input_prompts['input_ids'], whole_prompts['input_ids']):
            # Remove padding
            padding = torch.count_nonzero(whole == self.qall_tokenizer.pad_token_id)
            whole = whole[padding:]
            padding = torch.count_nonzero(input == self.qall_tokenizer.pad_token_id)
            input = input[padding:]

            # Remove the last logit (unnecessary, automatically added by the model)
            logit = logit[:-1]

            # Get from the logits just the ones corresponding to the actual generation (label)
            good_logit = logit[-(len(whole) - len(input)):]

            # Get the label
            good_label = whole[len(input):]

            loss = self.loss_fn(
                good_logit,
                good_label,
            )
            losses.append(loss.item())
        return losses