
import numpy as np
import torch
#
# Tables got following
#  https://github.com/viplix3/YOLO/blob/master/k-means.py
# adapting the config

TabAnchors_9_anchors =[[73,74]
 ,[139, 212]
 ,[216, 510]
 ,[271, 148]
 ,[290, 296]
 ,[343, 562]
 ,[528, 231]
 ,[540, 362]
 ,[560, 555]]

# 36 anchors

TabAnchors_36 =[[ 47,  49]
 ,[ 90, 100]
 ,[102, 241]
 ,[141, 128]
 ,[149, 271]
 ,[165, 562]
 ,[170, 389]
 ,[176, 192]
 ,[216, 587]
 ,[239, 286]
 ,[239, 124]
 ,[250, 434]
 ,[274, 589]
 ,[279, 180]
 ,[301, 345]
 ,[335, 266]
 ,[335, 491]
 ,[336, 604]
 ,[356, 134]
 ,[386, 387]
 ,[395, 209]
 ,[403, 583]
 ,[456, 270]
 ,[462, 571]
 ,[474, 430]
 ,[478, 330]
 ,[513, 591]
 ,[523, 148]
 ,[554, 227]
 ,[587, 387]
 ,[589, 181]
 ,[590, 457]
 ,[593, 533]
 ,[599, 264]
 ,[601, 611]
 ,[602, 321]]

TabAnchors=[[ 46,  57]
 ,[ 49,  29]
 ,[ 82, 165]
 ,[ 90,  85]
 ,[ 96, 295]
 ,[111, 205]
 ,[124, 257]
 ,[133, 154]
 ,[134, 357]
 ,[138, 576]
 ,[138, 122]
 ,[158, 209]
 ,[166, 275]
 ,[174, 402]
 ,[176,  96]
 ,[183, 526]
 ,[186, 614]
 ,[199, 167]
 ,[209, 326]
 ,[215, 580]
 ,[218, 477]
 ,[227, 402]
 ,[236, 614]
 ,[238, 251]
 ,[247, 486]
 ,[255, 349]
 ,[258, 562]
 ,[260, 122]
 ,[262, 431]
 ,[264, 619]
 ,[268, 303]
 ,[274, 167]
 ,[279, 514]
 ,[284, 570]
 ,[290, 476]
 ,[290, 624]
 ,[300, 359]
 ,[301, 206]
 ,[304, 603]
 ,[305, 520]
 ,[311, 441]
 ,[314, 577]
 ,[315, 619]
 ,[330, 527]
 ,[333, 480]
 ,[333, 254]
 ,[333, 624]
 ,[334, 338]
 ,[334, 572]
 ,[344, 295]
 ,[353, 609]
 ,[357, 406]
 ,[358, 511]
 ,[360, 564]
 ,[362, 137]
 ,[366, 626]
 ,[385, 572]
 ,[386, 211]
 ,[388, 625]
 ,[390, 470]
 ,[397, 357]
 ,[403, 537]
 ,[417, 588]
 ,[419, 624]
 ,[421, 401]
 ,[434, 267]
 ,[437, 502]
 ,[455, 580]
 ,[456, 198]
 ,[464, 328]
 ,[471, 632]
 ,[480, 411]
 ,[483, 606]
 ,[489, 161]
 ,[492, 545]
 ,[494, 491]
 ,[508, 236]
 ,[512, 610]
 ,[517, 280]
 ,[529, 571]
 ,[541, 439]
 ,[543, 352]
 ,[549, 611]
 ,[556, 533]
 ,[562, 478]
 ,[563, 390]
 ,[565, 316]
 ,[582, 626]
 ,[584, 222]
 ,[603, 187]
 ,[603, 150]
 ,[607, 259]
 ,[608, 410]
 ,[608, 508]
 ,[612, 589]
 ,[613, 552]
 ,[617, 305]
 ,[619, 353]
 ,[619, 459]
 ,[624, 619]]

# COPIED FROM  https://github.com/Megvii-BaseDetection/YOLOX/blob/ac58e0a5e68e57454b7b9ac822aced493b553c53/yolox/models/losses.py#L10C51-L10C51
def IOULoss(pred, target):
        reduction="none"
        loss_type="iou"
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        tl = torch.max(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )
        br = torch.min(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )

        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        area_u = area_p + area_g - area_i
        iou = (area_i) / (area_u + 1e-16)

        if loss_type == "iou":
            loss = 1 - iou ** 2
        elif loss_type == "giou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_u) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)

        if reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()

        return loss

def GetFit(w,h):
    lossMin=99999.0
    indxMin=-1
    # format loss array as required by  
    pred=torch.tensor([0,0,w,h])
    #pred=np.array(pred)
    for i in range (len(TabAnchors)):
        Wtarget=TabAnchors[i][0]
        Htarget=TabAnchors[i][1]
        loss=IOULoss(pred,torch.tensor([0,0,Wtarget,Htarget]))
        if loss < lossMin:
            lossMin=loss
            indxMin=i
    return lossMin,TabAnchors[indxMin][0], TabAnchors[indxMin][1]

# Main with test data
w=73
h=74
loss, w, h = GetFit(w,h)
print("loss "+ str(loss))
print("w "+ str(w))
print("h "+ str(h))


