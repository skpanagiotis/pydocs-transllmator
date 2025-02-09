import argparse
import time
import polib
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer

# Load MarianMT translation model for English to Greek
MODEL_NAME = "Helsinki-NLP/opus-mt-en-el"
tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
model = MarianMTModel.from_pretrained(MODEL_NAME)


def translate_text(text):
    if not text.strip():
        return text  # Skip empty strings

    inputs = tokenizer(text, return_tensors="pt",
                       padding=True, truncation=True)
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def translate_po_file(input_file, output_file):
    """Translate a .po file and save the translated version."""
    po = polib.pofile(input_file)
    print(len(po))
    progress = tqdm(total=len(po), desc="Translating po", unit='pofile')

    for entry in po:
        if not entry.translated():  # Only translate if msgstr is empty
            entry.msgstr = translate_text(entry.msgid)
        progress.update(1)

    po.save(output_file)
    progress.close()
    print(f"✅ Translated file saved as: {output_file}")


if __name__ == '__main__':
    start = time.time()
    # Usage Example
    translate_po_file("files/test.po", "files/translated.po")
    print(f"Translated in {time.time() - start}")
