import argparse
import logging
from argparse import Namespace
from transformers import AutoTokenizer
import csv
from fairseq.data import (
    Dictionary, 
    encoders, 
    PrependTokenDataset,
    AppendTokenDataset, 
    data_utils, 
    StripTokenDataset,
    TokenBlockDataset,
)
from fairseq.tasks.hubert_pretraining import LabelEncoder 

"""
Usage:
python scripts/validate_prepared.py ../../data/tsv/dev_other.txt ../../data/tsv/dev_other.tsv ../../my_models/spm_char.model ../../data/pretrain2/dict.txt
"""
logger = logging.getLogger(__name__)

def build_bpe(bpe_tokenizer):
    
 
    logger.info(f"tokenizer: {bpe_tokenizer}")
    return encoders.build_bpe(Namespace(**{"bpe": "sentencepiece", "sentencepiece_model": bpe_tokenizer}))
        
def validate(text_file, tsv_file, tokenizer_model, dict_):
    # Load the tokenizer
    bpe_tokenizer = build_bpe(tokenizer_model)
    dict_ = Dictionary.load(dict_)
    label_processors = LabelEncoder(dict_)
    # Read audio lengths from TSV file
    with open(tsv_file, "r") as tsv:
        reader = csv.reader(tsv, delimiter="\t")
        
        audio_lengths = [int(row[1]) for row in reader if len(row) > 1]
    
    # Read transcripts from text file
    with open(text_file, "r") as txt:
        transcripts = [line.strip() for line in txt]
   
    if len(audio_lengths) != len(transcripts):
        raise ValueError("Mismatch between audio lengths and text lines")

    # Iterate and compare tokens
    for line_num, (audio_len, transcript) in enumerate(zip(audio_lengths, transcripts), start=1):
        # Tokenize transcript using the tokenizer
        
        letter_tokens = bpe_tokenizer.encode(transcript)
        label = label_processors(letter_tokens)
       
        # Downsample audio length
        downsampled_audio_tokens = audio_len // 320
        
        # Compare and print if downsampled audio is smaller
        if downsampled_audio_tokens < len(label):
            print(f"Line {line_num}:")
            print(f"  Downsampled audio tokens: {downsampled_audio_tokens}")
            print(f"  Letter tokens: {len(label)}")
            print(f"  Transcript: {transcript}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate text and audio data.")
    parser.add_argument("text_file", type=str, help="Path to the text file containing tokenized transcripts")
    parser.add_argument("tsv_file", type=str, help="Path to the TSV file containing audio lengths.")
    parser.add_argument("tokenizer_model", type=str, help="Path or name of the tokenizer model.")
    parser.add_argument("dict", type=str, help="Path to the dictionary.")

    args = parser.parse_args()

    validate(args.text_file, args.tsv_file, args.tokenizer_model, args.dict)

