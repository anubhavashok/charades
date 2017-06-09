import torch
import matplotlib.pyplot as plt
import numpy as np
import itertools
from torch.autograd import Variable

def top5acc(pred, target):
    pred = pred.cpu()
    target = target.cpu()
    _, i = torch.topk(pred, 5, dim=1)
    i = i.type_as(target)
    mn, _ = torch.max(i.eq(target.repeat(5, 1).t()), dim=1)
    acc = torch.mean(mn.float())
    return acc

def resetModel(m):
    if len(m._modules) == 0 and hasattr(m, 'reset_parameters'):
        m.reset_parameters()
        return
    for i in m._modules.values():
        resetModel(i)

def clip_grad(v, min, max):
    v.register_hook(lambda g: g.clamp(min, max))
    return v

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    #for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #    plt.text(j, i, cm[i, j],
    #             horizontalalignment="center",
    #             color="white" if cm[i, j] > thresh else "black")

    #plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('cmatrix.png')


def writeTestScore(f, vid, scores):
    # perform merging algorithm
    score = scores[0].data.clone().fill_(0)
    k = 0
    for i in range(len(scores)):
        _, j = torch.max(scores[i], 0)
        if j != 157:
            score += scores[i].data
            k += 1
    score /= k
    score = score.cpu().numpy().tolist()[:-1]
    score = scores[-1].data.cpu().numpy().tolist()[:-1]
    f.write("%s %s\n" % (vid, ' '.join(map(str, score))))


def one_hot(size, index):
    """ Creates a matrix of one hot vectors.
        ```
        import torch
        import torch_extras
        setattr(torch, 'one_hot', torch_extras.one_hot)
        size = (3, 3)
        index = torch.LongTensor([2, 0, 1]).view(-1, 1)
        torch.one_hot(size, index)
        # [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
        ```
    """
    index = index.long().cpu()
    index = index.unsqueeze(1)
    mask = torch.LongTensor(*size).fill_(0)
    ones = 1
    if isinstance(index, Variable):
        ones = Variable(torch.LongTensor(index.size()).fill_(1))
        mask = Variable(mask, volatile=index.volatile)
    ret = mask.scatter_(1, index, ones)
    return ret.cpu().numpy()

