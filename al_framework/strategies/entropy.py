import math

def sum_entropy(al_cfg, predictions, k):
    topScores = []
    for pred in predictions:
        filename, softmaxes = pred[0], pred[1]
        softmaxScores = []
        if softmaxes:
            for softmax in softmaxes:
                softmaxScore = 0
                for prob in softmax:
                    if float(prob) != 0.0:
                        softmaxScore -= float(prob)*math.log(float(prob), al_cfg.MODEL.NUM_CLASSES)
                    else:
                        prob = 0.00000001
                        softmaxScore -= float(prob)*math.log(float(prob), al_cfg.MODEL.NUM_CLASSES)
                softmaxScores.append(softmaxScore)
        else:
            softmaxScores.append(-1)
        topScores.append([filename, sum(softmaxScores)])
    topScores = sorted(topScores, key=lambda x: x[1], reverse=True)
    return topScores[:k]

def max_entropy(al_cfg, predictions, k):
    topScores = []
    for pred in predictions:
        filename, softmaxes = pred[0], pred[1]
        softmaxScores = []
        if softmaxes:
            for softmax in softmaxes:
                softmaxScore = 0
                for prob in softmax:
                    if float(prob) != 0.0:
                        softmaxScore -= float(prob)*math.log(float(prob), al_cfg.MODEL.NUM_CLASSES)
                    else:
                        prob = 0.00000001
                        softmaxScore -= float(prob)*math.log(float(prob), al_cfg.MODEL.NUM_CLASSES)
                softmaxScores.append(softmaxScore)
        else:
            softmaxScores.append(-1)
        topScores.append([filename, max(softmaxScores)])
    topScores = sorted(topScores, key=lambda x: x[1], reverse=True)
    return topScores[:k]

def avg_entropy(al_cfg, predictions, k):
    topScores = []
    for pred in predictions:
        filename, softmaxes = pred[0], pred[1]
        softmaxScores = []
        if softmaxes:
            for softmax in softmaxes:
                softmaxScore = 0
                for prob in softmax:
                    if float(prob) != 0.0:
                        softmaxScore -= float(prob)*math.log(float(prob), al_cfg.MODEL.NUM_CLASSES)
                    else:
                        prob = 0.00000001
                        softmaxScore -= float(prob)*math.log(float(prob), al_cfg.MODEL.NUM_CLASSES)
                softmaxScores.append(softmaxScore)
        else:
            softmaxScores.append(-1)
        topScores.append([filename, sum(softmaxScores)/len(softmaxScores)])
    topScores = sorted(topScores, key=lambda x: x[1], reverse=True)
    return topScores[:k]
