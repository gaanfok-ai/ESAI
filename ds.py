from datasets import load_dataset

def normalize_labels(labels, texts, answerKey):
    proper_labels = ['A', 'B', 'C', 'D', 'E']
    answer_text = None
    for i in range(len(labels)):
        if answerKey == labels[i]:
            answerKey = proper_labels[i]
            answer_text = texts[i]
        labels[i] = proper_labels[i]
    return labels, answer_text, answerKey
        
def format_opbqa(ex):
    id = ex['id']
    question = ex['question_stem']
    choices_text = ex['choices']['text']
    labels, answer_text, answer_key = normalize_labels(ex['choices']['label'], ex['choices']['text'], ex['answerKey'])
    choices = "\n".join([f"{l}. {c}" for l, c in zip(labels, choices_text)])
    return {
        "id": id,
        "question": question,
        "choices": choices,
        "answerKey": answer_key,
        "answerText": answer_text,
    }

def format_arc(ex):
    id = ex['id']
    question = ex['question']
    choices_text = ex['choices']['text']
    labels, answer_text, answer_key = normalize_labels(ex['choices']['label'], ex['choices']['text'], ex['answerKey'])
    choices = "\n".join([f"{l}. {c}" for l, c in zip(labels, choices_text)])
    return {
        "id": id,
        "question": question,
        "choices": choices,
        "answerKey": answer_key,
        "answerText": answer_text,
    }

def load_and_format_opbqa(split="test"):
    print("Loading OpenbookQA dataset...")
    ds = load_dataset("allenai/openbookqa", "main", split=split)
    ds_formatted = ds.map(format_opbqa, remove_columns=ds.column_names)
    print(f"Openbookqa loaded! Length: {len(ds_formatted)}")
    return ds_formatted

def load_and_format_arc(diff="ARC-Challenge", split="test"):
    print(f"Loading {diff}...")
    ds = load_dataset("allenai/ai2_arc", diff, split=split)
    ds_formatted = ds.map(format_arc, remove_columns=ds.column_names)
    print(f"{diff} loaded! Length {len(ds_formatted)}")
    return ds_formatted

def main():
    opbqa = load_and_format_opbqa()
    arc_easy = load_and_format_arc(diff='ARC-Easy')
    arc_challenge = load_and_format_arc(diff='ARC-Challenge')
    print(len(opbqa), len(arc_easy), len(arc_challenge))
    print("Openbookqa structure:\n", opbqa)
    print(opbqa[0]['answerText'])

if __name__ == "__main__":
    main()
