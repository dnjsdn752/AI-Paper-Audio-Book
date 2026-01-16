from transformers import AutoProcessor, AutoModelForCausalLM
'''
processor = AutoProcessor.from_pretrained("microsoft/git-base")
model = AutoModelForCausalLM.from_pretrained("microsoft/git-base")

processor.save_pretrained("./model/microsoft_git_base")
model.save_pretrained("./model/microsoft_git_base")