import torch
import matplotlib.pyplot as plt
import numpy as np
import itertools
from torch.autograd import Variable
from torch.nn.modules.loss import _WeightedLoss
from torch.nn import MSELoss

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

def removeEmptyFromTensor(input, target):
    mask = target.sum(1) > 0
    mask = mask.squeeze().nonzero().squeeze()
    target = target.index_select(0, mask)
    input = (input[0].index_select(0, mask), input[1].index_select(0, mask))
    return input, target

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

classweights = [0.015058585786963999, 0.010392108202010175, 0.008189601687554287, 0.003952989559144169, 0.007719851807207558, 0.0029514473614237853, 0.005583819332423377, 0.0008685941183769699, 0.006860120894120149, 0.007431797635296829, 0.0014358392569088685, 0.01815627603566554, 0.0050608902203392835, 0.0008863205289560917, 0.008238349316646873, 0.022667647528052046, 0.02088171166220552, 0.002038537216599011, 0.004068211227908461, 0.004533529505610409, 0.0141057912183362, 0.004019463598815875, 0.003518692499955684, 0.0038023150692216333, 0.0006026979596901424, 0.0012940279722758938, 0.012102706822895432, 0.002158190488008083, 0.0027342988318295428, 0.0011167638664846755, 0.003075532235477638, 0.00027475936397638843, 0.010369950188786272, 0.012656657153492989, 0.002601350752486129, 0.002814067679435591, 0.0009749525818517009, 0.002211369719745449, 0.0063637813979047385, 0.0012630067537624306, 0.006301738960877812, 0.001989789587506426, 0.002619077163065251, 0.0025658979313278856, 0.001777072660556964, 0.00023930654281814476, 0.0003633914168719976, 0.004772836048428554, 0.0006558771914275079, 0.0009660893765621399, 0.0010458582241681881, 0.007077269423714392, 0.007941431939446582, 0.004046053214684558, 0.0018789995213869144, 0.003128711467215004, 0.002167053693297644, 0.002862815308528176, 0.0006647403967170687, 0.036906386825731656, 0.001333912396078918, 0.030405225745838725, 0.006979774165529222, 0.009226596706432914, 0.0006337191782036056, 0.0022069381171006684, 0.00034566500629287577, 0.005681314590608548, 0.0009838157871412618, 0.001218690727314626, 0.010622551539538758, 0.002490560686366618, 0.007578040522574584, 0.002619077163065251, 0.0007046248205200929, 0.003704819811036463, 0.006913300125857515, 0.001874567918742134, 0.002379770620247106, 0.0018302518922943293, 0.0011655114955772606, 0.005752220232925035, 0.007604630138443267, 0.001130058674419017, 0.0022379593356141314, 0.00018612731108077926, 0.0005406555226632159, 0.0028849733217520784, 0.005158385478524453, 0.001169943098222041, 0.0017061670182404766, 0.0007312144363887756, 0.007937000336801801, 0.0021404640774289616, 0.0008065516813500434, 0.0004653182777019481, 0.010941626929962952, 0.023159555421622676, 0.006151064470955276, 0.0009262049527591158, 0.002109442858915498, 0.00018612731108077926, 0.0035895981422721713, 0.001267438356407211, 0.002158190488008083, 0.0019499051637034018, 0.010392108202010175, 0.02543296757839505, 0.004037190009394997, 0.005623703756226402, 0.006470139861379469, 0.0011655114955772606, 0.005176111889103575, 0.006984205768174002, 0.007072837821069612, 0.01160193572403524, 0.0015776505415418432, 0.0021227376668498396, 0.02278730079946112, 0.006053569212770106, 0.005809831067307181, 0.003518692499955684, 0.00280520447414603, 0.01188998989594597, 0.0057300622197011325, 0.019078049385779873, 0.0034211972417705137, 0.017642210128871006, 0.006430255437576445, 0.002268980554127595, 0.0025658979313278856, 0.0003633914168719976, 0.011442398028823143, 0.0025304451101696417, 0.006239696523850886, 0.009044900997996916, 0.0010857426479712123, 0.004763972843138993, 0.0017637778526226225, 0.002880541719107298, 0.001112332263839895, 0.006975342562884441, 0.0015865137468314041, 0.0033104071756510024, 0.0036250509634304148, 0.0030976902487015404, 0.0032084803148210517, 0.014150107244784004, 0.01005530640100686, 0.008561856309715846, 0.005517345292751671, 0.008947405739811745, 0.020558204669136545, 0.009346249977841987, 0.014965522131423608, 0.008340276177476822, 0.008402318614503749, 0.10166982787655328]



invclassweights = [1/ii for ii in classweights]
invclassweights = [ii/sum(invclassweights) for ii in invclassweights]
invClassWeightstensor = torch.FloatTensor(invclassweights)


class TripletLoss(_WeightedLoss):
    def __init__(self):
        super(TripletLoss, self).__init__()
        self.mseLoss = MSELoss()
        self.alpha = 1
    
    def forward(self, inp, positive, negative):
        loss = self.mseLoss(inp, positive) - self.mseLoss(inp, negative) + self.alpha
        return loss

