from transformers import BertTokenizer, BertForSequenceClassification

def get_model_and_tokenizer():
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name)
    return model, tokenizer
