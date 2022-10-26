import torch
import numpy as np
from poly_iou import poly_iou
import os
from tqdm import tqdm

def ap_per_class(tp, conf, pred_cls, target_cls, names=[], eps=1e-16):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    names=dict(zip(range(len(names)),names))
    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # number of labels
        n_p = i.sum()  # number of predictions

        if n_p == 0 or n_l == 0:
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_l + eps)  # recall curve
            r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + eps)
    names = [v for k, v in names.items() if k in unique_classes]  # list: only classes that have data
    names = {i: v for i, v in enumerate(names)}  # to dict
    i = f1.mean(0).argmax()  # max F1 index
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    return tp, fp, p, r, f1, ap, unique_classes.astype('int32')



def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec
  
  
def process_batch_poly(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x, y, w, h, θ) format.
    Arguments:
        detections (Array[N, 7]), x, y, w, h, θ, class, conf
        labels (Array[M, 7]),  x, y, w, h, θ, class, difficult
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    iou = poly_iou(labels[:, :5], detections[:, :5])
    x = torch.where((iou >= iouv[0]) & (labels[:, -2:-1] == detections[:, -2]))  # IoU above threshold and classes match
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detection, iou]
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            # matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.Tensor(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    return correct



class Eval:
    def __init__(self, names,device="cuda"):
        self.device=device
        self.iouv=torch.linspace(0.5, 0.95, 10).to(device)
        self.names=names
        self.nc=len(names)
        self.stats=[]
        self.ap=[]
        self.ap_class = []
        self.images=0
        self.ap50,self.r,self.p,self.nt,self.mp, self.mr,self.map50, self.map = None, None, None, None, None,None,None,None
        
    def append_result(self,ppoly,tpoly):
        self.images+=1
        nl=len(tpoly)
        tcls=tpoly[:,-2].cpu().tolist() if nl else []
        if len(ppoly)==0:
            if nl:
                self.stats.append((torch.zeros(0, self.iouv.numel(), dtype=torch.bool), torch.Tensor(),  torch.Tensor(), tcls))
        else:
            if nl:
                correct = process_batch_poly(ppoly, tpoly, self.iouv)
            
            else:
                correct = torch.zeros(ppoly.shape[0], self.iouv.numel(), dtype=torch.bool)
        self.stats.append((correct.cpu(), ppoly[:,-1].cpu(), ppoly[:,-2].cpu(), tcls))
        
    def compute_metrics(self,verbose=True):
      stats = [np.concatenate(x, 0) for x in zip(*(self.stats))] 
      if len(stats) and stats[0].any():
          tp, fp,self.p, self.r, f1, self.ap, self.ap_class = ap_per_class(*stats,names=self.names)
          self.ap50, self.ap = self.ap[:, 0], self.ap.mean(1)
          self.mp, self.mr,self.map50, self.map = self.p.mean(), self.r.mean(),self.ap50.mean(), self.ap.mean()
          self.nt = np.bincount(stats[3].astype(np.int64), minlength=self.nc)
      else:
          self.nt = torch.zeros(1)
      
      if verbose:
          self.print_log()
          
      return self.ap_class
    
    def print_log(self):
        print(('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', '  mAP@.5:.95'))
        pf = '%20s' + '%11i' * 2 + '%11.3g' * 4  # print format
        print(pf % ('all', self.images, self.nt.sum(), self.mp, self.mr, self.map50, self.map))
        # Print results per class
        if  (self.nc < 50 ) and self.nc > 1 and len(self.stats):
            for i, c in enumerate(self.ap_class):
                print(pf % (self.names[c], self.images, self.nt[c], self.p[i], self.r[i], self.ap50[i], self.ap[i]))  
          
    def parse_file(self,gt_path,dt_path):
        index_map=dict(zip(self.names,range(self.nc)))
        gt_file_list=os.listdir(gt_path)
        dt_file_list=os.listdir(dt_path)
        assert gt_file_list is not None,"Groungtruth is None,Plase check path"
        assert dt_file_list is not None,"Pred results is None,Plase check path"
        skip_list=[]
        for i in gt_file_list:
          if i not in dt_file_list:
            skip_list.append(i)
        for i in tqdm(dt_file_list):
            with open(os.path.join(dt_path,i))  as fp:
                pred_results=fp.readlines()
            with open(os.path.join(gt_path,i))  as fp:
                targets_results=fp.readlines()
            predlines = [x.strip().split() for x in pred_results]
            targetlines = [x.strip().split() for x in targets_results]
            for index,value in enumerate(targetlines) :
                targetlines[index][-2]=index_map[value[-2]]
            for index,value in enumerate(predlines) :
                predlines[index][-2]=index_map[value[-2]]
            pred=np.array(predlines,dtype=np.float32)
            target=np.array(targetlines,dtype=np.float32)
            pred=torch.from_numpy(pred).cuda()
            target=torch.from_numpy(target).cuda()
            self.append_result(pred,target)
            
if __name__ == "__main__":
    names=["car","truck","bus","van","freight_car"]
    myeval=Eval(names)
    # gt_path="/home/eeg/Project1/cgc/datasets/VisDrone-DroneVehicle/poly/val/labels"
    pred_path="/home/cv/Project/cgc/Det220/test_rbox/yolov5_obb/runs/val/exp3/labels"
    
    gt_path="/home/cv/Project/cgc/Det220/test_rbox/val/labels"
    # pred_path="/home/eeg/Project1/cgc/yolov5_obb/runs/val/exp6/test"
    
    print("---------Start Statistics--------")
    myeval.parse_file(gt_path,pred_path)
    print("------Start Compute metrics------")
    myeval.compute_metrics()