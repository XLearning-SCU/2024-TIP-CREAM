import math
import copy
import torch
import torch.nn as nn


class InfoNCELoss(nn.Module):
    """
    Compute infoNCE loss
    """
    def __init__(self, temperature=0.07):
        super(InfoNCELoss, self).__init__()

        self.temperature = temperature


    def forward(
        self,
        scores,# similarity matrix.
        labels=None, # refined labels. when mode="predict", labels must be None.
        mode="train",
        soft_label=True,   # when true, one hot is changed to soft during calculating infoNCE loss 
        smooth_label=False, # this argument is useless when soft_label is set to False
        filter=0, # if weight < filter, then weight = 0, only for soft_label=True, smooth_label=False
    ):  
        if labels is not None:
            labels = labels.t()[0]
        similarity = scores.clone() # copy a similarity as backup.
        scores = torch.exp(torch.div(scores, self.temperature))
        diagonal = scores.diag().view(scores.size(0), 1).t().to(scores.device)
        
        sum_row = scores.sum(1)
        sum_col = scores.sum(0)
        
        loss_text_retrieval = -torch.log(torch.div(diagonal, sum_row))
        loss_image_retrieval = -torch.log(torch.div(diagonal, sum_col))
        
        if mode == "predict":

            predict_text_retrieval = torch.div(scores.t(), sum_row).t()
            predict_image_retrieval = torch.div(scores, sum_col)

            p = (predict_text_retrieval + predict_image_retrieval) / 2
            p = p.diag().view(scores.size(0), 1).t().to(scores.device)
            p = p.clamp(min=0, max=1)
            
            return p
            
        elif mode == "warmup":
            # Warm up with the original infoNCE.
            return (loss_text_retrieval + loss_image_retrieval)[0].mean(), torch.Tensor([0]), torch.Tensor([0])
        elif mode == "train":
            if soft_label:
                # soft_label, two versions.
                loss_text_retrieval = -torch.log(torch.div(scores.t(), sum_row).t())
                loss_image_retrieval = -torch.log(torch.div(scores, sum_col))
                
                if smooth_label:
                    # positive pairs' weights are labels, and the remaining 
                    # sample pairs have mean scores: (1-labels) / batch_size - 1.  
                    label_smooth = ((1 - labels)/(scores.size(0)-1)) * torch.ones(scores.size()).to(scores.device)
                    labels = torch.eye(scores.size(0)).to(labels.device) * labels
                    # if labels = torch.tensor([0.7,0.8,0.9,0.4]), size is (1,4)
                    # then the sum of label_smooth‘s each column is one

                    mask = torch.eye(scores.size(0)) > 0.5
                    mask = mask.to(label_smooth.device)
                    label_smooth = label_smooth.masked_fill_(mask, 0)
                    label_smooth = labels + label_smooth


                    loss_text_retrieval = torch.mul(label_smooth.t(), loss_text_retrieval)
                    loss_image_retrieval = torch.mul(label_smooth, loss_image_retrieval)

                    return (loss_text_retrieval + loss_image_retrieval).sum() / scores.size(0), (torch.max(label_smooth + label_smooth.t() - 2 * labels, 1)[0].mean() / 2), (torch.max(label_smooth + label_smooth.t() - 2 * labels, 0)[0].mean() / 2)
                
                else:
                    # the remaining sample pairs share (1-labels) according to similarity (scores).
                    positive_weight = torch.eye(scores.size(0)).to(labels.device) * labels
                    mask = torch.eye(scores.size(0)) > 0.5
                    mask = mask.to(similarity.device)
                    similarity = similarity.masked_fill_(mask, 0)

                    sum_row = similarity.sum(1)
                    sum_col = similarity.sum(0)

                    # add a very small value (1e-6) to prevent division by 0.
                    weight_row = torch.div(similarity.t(), sum_row + 1e-6).t()
                    weight_col = torch.div(similarity, sum_col + 1e-6)

                    d1 = (1-labels).expand_as(weight_row).t()
                    d2 = (1-labels).expand_as(weight_col)
                    weight_row = torch.mul(d1, weight_row)
                    weight_col = torch.mul(d2, weight_col)

                    weight_col[weight_col < filter] = 0
                    weight_row[weight_row < filter] = 0

                    weight_row = weight_row + positive_weight
                    weight_col = weight_col + positive_weight
                    
                    loss_text_retrieval = torch.mul(loss_text_retrieval, weight_row)
                    loss_image_retrieval = torch.mul(loss_image_retrieval, weight_col)

                    return ((loss_text_retrieval + loss_image_retrieval).sum() / scores.size(0)), (torch.max(weight_row + weight_col - 2 * positive_weight, 1)[0].mean() / 2), (torch.max(weight_row + weight_col - 2 * positive_weight, 0)[0].mean() / 2)

            else:
                return (loss_text_retrieval + loss_image_retrieval)[0].mean(), torch.Tensor([0]), torch.Tensor([0])

        # two versions for eval loss. one is using the positive pair for loss computation, the other is using infoNCE loss for per sample loss computation.
        elif mode == "eval_loss":
            # using infoNCE loss for per sample loss computation.
            return (loss_text_retrieval + loss_image_retrieval)[0], torch.Tensor([0]), torch.Tensor([0])

        # only using positive part for computation.
        # elif mode == "eval_loss":
        #     return -torch.log(diagonal)[0], torch.Tensor([0]), torch.Tensor([0])
