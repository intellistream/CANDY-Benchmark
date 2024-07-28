from typing import List
import torch

def exist_row(base:torch.tensor, row:torch.tensor)->bool:
    for i in range(base.size(0)):
        tensor1 = base[i]
        tensor2= row

        if(torch.equal(tensor1, tensor2)):
            return True
        
    return False


def calculate_recall(ground_truth:List[torch.tensor], prob:List[torch.tensor]) -> float:
    true_positives = 0
    false_negatives = 0
    aknn = len(prob)
    for i in range(len(prob)):
        gdI = ground_truth[i]
        probI = prob[i]
        for j  in range(probI.size(0)):
            if(exist_row(gdI, probI[j])):
                true_positives +=1
            else:
                false_negatives+=1
    recall = true_positives / (true_positives+false_negatives)
    return recall    



gd = [torch.Tensor([[1,2,3,4,5],[2,3,4,5,6],[3,4,5,6,7]]), torch.Tensor([[1,3,5,7,9],[2,4,6,8,10],[4,6,8,10,12]])]
prob = [torch.Tensor([[0,0,0,0,0],[1,2,3,4,5],[1,5,67,78,1]]), torch.Tensor([[0,0,0,0,0],[4,6,8,10,12],[2,4,6,8,10]])]

print(calculate_recall(gd, prob))