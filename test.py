from collections import Counter
import string
import re
import argparse
import json
import sys
from question_answering import QuestionAnsweringModel, QuestionAnsweringArgs

def convert_data(data):
    converted = []
    for d in data:
        t = {}
        t['context'] = d['paragraphs'][0]['context']
        qas = []
        for q in d['paragraphs'][0]['qas']:
            q['is_impossible'] = False
            qas.append(q)
        t['qas'] = qas
        converted.append(t)
    return converted


def read_data(PATH):
    with open(PATH+'train.json', 'r', encoding='utf8') as f:
        TRAIN = json.load(f)

    with open(PATH+'dev.json', 'r', encoding='utf8') as f:
        DEV = json.load(f)

    with open(PATH+'test.json', 'r', encoding='utf8') as f:
        TEST = json.load(f)

    return TRAIN, DEV, TEST

def normalize_answer(s):
	"""Lower text and remove punctuation, articles and extra whitespace."""

	def white_space_fix(text):
		return ' '.join(text.split())

	def remove_punc(text):
		exclude = set(string.punctuation)
		return ''.join(ch for ch in text if ch not in exclude)

	def lower(text):
		return text.lower()

	return white_space_fix(remove_punc(lower(s)))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate(dataset, predictions):
    f1 = exact_match = total = 0
    for article in dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                total += 1
                if qa['id'] not in predictions:
                    message = 'Unanswered question ' + qa['id'] + \
                              ' will receive score 0.'
                    print(message, file=sys.stderr)
                    continue
                ground_truths = list(map(lambda x: x['text'], qa['answers']))
                prediction = predictions[qa['id']]
                exact_match += metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths)
                f1 += metric_max_over_ground_truths(
                    f1_score, prediction, ground_truths)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {'exact_match': exact_match, 'f1': f1}


def load_model(type, path):
    model = QuestionAnsweringModel(
        type, path
    )

    return model


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default="/home/data")
    parser.add_argument('--type', type=str, default="auto")
    parser.add_argument('--output_path', type=str, default="/home/data")
    parser.add_argument('--is_test', default=False, action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parser_args()
    train, dev, test = read_data(args.path)

    train_squad = convert_data(train['data'])
    dev_squad = convert_data(dev['data'])
    test_squad = convert_data(test['data'])

    model = load_model(args.type, args.output_path)

    if args.is_test:
        print('--test--')
        result, model_outputs = model.eval_model(test_squad, output_dir=args.output_path)
        with open(args.output_path + '/predictions_test.json', 'r', encoding='utf-8') as f:
            pred = json.load(f)

        print(evaluate(test['data'], pred))
    else:
        print('--dev--')
        result, model_outputs = model.eval_model(dev_squad, output_dir=args.output_path)
        with open(args.output_path + '/predictions_test.json', 'r', encoding='utf-8') as f:
            pred = json.load(f)

        print(evaluate(dev['data'], pred))
