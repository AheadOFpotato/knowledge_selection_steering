# Official checkanswer from chen700564/RGB evalue.py (isolated; no model deps).
# Source: https://github.com/chen700564/RGB/blob/master/evalue.py


def checkanswer(prediction, ground_truth):
    prediction = prediction.lower()
    if type(ground_truth) is not list:
        ground_truth = [ground_truth]
    labels = []
    for instance in ground_truth:
        flag = True
        if type(instance) == list:
            flag = False
            instance = [i.lower() for i in instance]
            for i in instance:
                if i in prediction:
                    flag = True
                    break
        else:
            instance = instance.lower()
            if instance not in prediction:
                flag = False
        labels.append(int(flag))
    return labels
