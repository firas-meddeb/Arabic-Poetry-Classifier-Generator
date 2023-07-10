from transformers import GPT2Tokenizer, TFGPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained('akhooli/gpt2-small-arabic', from_pt = True)
model = TFGPT2LMHeadModel.from_pretrained('akhooli/gpt2-small-arabic', pad_token_id = tokenizer.eos_token_id, from_pt = True)
text = 'الثقافة هي سلوك اجتماعي ومعيار'
input_ids = tokenizer.encode(text, return_tensors = 'pt')
greedy_output = model.generate(input_ids, max_length = 50)
s = tokenizer.decode(greedy_output[0], skip_special_tokens = True)
with open('output.txt', 'w') as file:
    file.writelines(s)