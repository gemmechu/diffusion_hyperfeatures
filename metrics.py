import torch

def px_th(gt, pred, query_pts, visibles=None, th=[1, 2, 4, 8, 16]):
    """
    End-point-error accuracy
    
    gt (torch.Tensor): B x num_pts x num_frames x 2 (x, y)
    pred (torch.Tensor): B x num_pts x num_frames x 2 (x, y) -- directly out of model
    query_pts (torch.Tensor): B x num_pts x 3 (t, y, x)
    th (List[float]) thresholds
    """
    
    B, num_query_pts, num_frames, _ = gt.shape
    # remove the queried frame from the tracks
    # TAPIR usually predicts them with high accuracy
    # skewing the results
    non_query_mask = (
        1 - F.one_hot(
                query_pts[..., 0].long(), 
                num_classes=num_frames
            )
        ).bool()
    gt = gt[..., non_query_mask, :].reshape(B, num_query_pts, -1, 2)
    pred = pred[..., non_query_mask, :].reshape(B, num_query_pts, -1, 2)
    
    
    num_vids =  B
    N = gt.shape[1] * gt.shape[2]
    if visibles is not None:
        visibles = visibles[..., non_query_mask].reshape(B, num_query_pts, -1)
        pred = pred[visibles]
        gt = gt[visibles]
        
        
    diff = pred - gt
    diff_norm = torch.norm(diff, p=2, dim=-1)
    diff_norm = diff_norm.reshape(-1)
    acc_at_th = dict()
    
    for t in th:
        num_acc = (diff_norm < t).int().sum().item()
        num_tot = N
        acc_at_th[f"epea_{int(t)}"] = num_acc / num_tot / num_vids
        # acc_at_th.append(num_acc / num_tot / num_batches)
    
    acc_at_th['mean_epea'] = sum(list(acc_at_th.values())) / len(acc_at_th) 
    return acc_at_th


def px_th_noquery(gt, pred, visibles=None, th=[1, 2, 4, 8, 16]):
    """
    End-point-error accuracy
    
    gt (torch.Tensor): B x num_pts x num_frames (1) x 2 (x, y)
    pred (torch.Tensor): B x num_pts x num_frames (1) x 2 (x, y) -- directly out of model
    th (List[float]) thresholds
    """
    
    B, num_pts, num_frames, _ = gt.shape
    
    num_vids =  B
    N = gt.shape[1] * gt.shape[2]
    if visibles is not None:
        visibles = visibles[..., non_query_mask].reshape(B, num_query_pts, -1)
        pred = pred[visibles]
        gt = gt[visibles]
        
        
    diff = pred - gt
    diff_norm = torch.norm(diff, p=2, dim=-1)
    diff_norm = diff_norm.reshape(-1)
    acc_at_th = dict()
    
    for t in th:
        num_acc = (diff_norm < t).int().sum().item()
        num_tot = N
        acc_at_th[f"epea_{int(t)}"] = num_acc / num_tot / num_vids
        # acc_at_th.append(num_acc / num_tot / num_batches)
    
    acc_at_th['mean_epea'] = sum(list(acc_at_th.values())) / len(acc_at_th) 
    return acc_at_th
