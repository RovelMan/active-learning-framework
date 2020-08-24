
def sum_bvsb(predictions, k):
    topScores = []
    for pred in predictions:
        filename, softmaxes = pred[0], pred[1]
        marginScores = []
        if softmaxes:
            for softmax in softmaxes:
                softmax = [float(s) for s in softmax]
                softmax.sort()
                marginScores.append(1-softmax[-1]-softmax[-2])
        else:
            marginScores.append(-1)
        topScores.append([filename, sum(marginScores)])
    topScores = sorted(topScores, key=lambda x: x[1], reverse=True)
    return topScores[:k]

def max_bvsb(predictions, k):
    topScores = []
    for pred in predictions:
        filename, softmaxes = pred[0], pred[1]
        marginScores = []
        if softmaxes:
            for softmax in softmaxes:
                softmax = [float(s) for s in softmax]
                softmax.sort()
                marginScores.append(1-softmax[-1]-softmax[-2])
        else:
            marginScores.append(-1)
        topScores.append([filename, max(marginScores)])
    topScores = sorted(topScores, key=lambda x: x[1], reverse=True)
    return topScores[:k]

def avg_bvsb(predictions, k):
    topScores = []
    for pred in predictions:
        filename, softmaxes = pred[0], pred[1]
        marginScores = []
        if softmaxes:
            for softmax in softmaxes:
                softmax = [float(s) for s in softmax]
                softmax.sort()
                marginScores.append(1-softmax[-1]-softmax[-2])
        else:
            marginScores.append(-1)
        topScores.append([filename, sum(marginScores)/len(marginScores)])
    topScores = sorted(topScores, key=lambda x: x[1], reverse=True)
    return topScores[:k]
