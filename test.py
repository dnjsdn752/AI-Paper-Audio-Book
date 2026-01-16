
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import deepl
import time

start = time.time()
print(f'{start}')
auth_key = "77a59737-132b-4dcf-86a8-2a8a6888c1c0:fx"
model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-coco")
processor = AutoProcessor.from_pretrained("microsoft/git-base-coco")
point1 = time.time()
print(f"{point1-start:.5f}sec")
def no_api_captioning(image):

    pixel_values = processor(images=image, return_tensors="pt").pixel_values

    generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    caption_text = generated_caption
    print(caption_text)
    translator = Translator()
    captioning_text = (translator.translate(caption_text, dest='ko').text)
    
    return captioning_text

def api_captioning(image):

    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    point2 = time.time()
    print(f"processor{point2-point1:.5f}sec")

    generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
    point3 = time.time()
    print(f"generate{point3-point2:.5f}sec")

    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    point4 = time.time()
    print(f"batch_decode{point4-point3:.5f}sec")

    caption_text = generated_caption
    point5 = time.time()
    print(f"generated_caption{point5-point4:.5f}sec")
    print(caption_text)

    deep_translator = deepl.Translator(auth_key)
    point6 = time.time()
    print(f"Translator{point6-point5:.5f}sec")

    captioning_text = (deep_translator.translate_text(caption_text, target_lang="ko").text)
    point7 = time.time()
    print(f"translate_text{point7-point6:.5f}sec")
    
    return captioning_text

file_path = './download.jpg'
image = Image.open(file_path)
caption = api_captioning(image)
print(caption)
end = time.time()
print(f"{end-start: .5f}sec")