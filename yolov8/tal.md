### 标签分配策略
#### yolov8版本为`__version__ = '8.0.110'`
* TaskAlignedAssigner简介

TaskAlignedAssigner 的匹配策略简单总结为：根据分类与回归的分数加权的分数选择正样本。

(1) 计算真实框和预测框的匹配程度。

$$align\\_metric =s^\alpha * u^\beta$$
 
其中，s是预测类别分值，u是预测框和真实框的ciou值，$`\alpha`$ 和$`\beta`$为权重超参数，两者相乘就可以衡量匹配程度，当分类的分值越高且ciou越高时，align_metric的值就越接近于1,此时预测框就与真实框越匹配，就越符合正样本的标准。

(2) 对于每个真实框，直接对align_metric匹配程度排序，选取topK个预测框作为正样本。

(3) 对一个预测框与多个真实框匹配测情况进行处理，保留ciou值最大的真实框。

* 代码实现流程
  1. 首先筛选锚点(特征图grid的坐标中心点)落在gt_box中, 得到mask_in_gt((Tensor): shape(b, n_boxes, h*w)), 其中1代表锚点落在gt_box中, 0表示锚点未落在gt_box中
  2. 计算匹配程度
     
     得到mask_gt,  mask_gt = mask_in_gt * mask_gt
     
     得到bbox_scores, 构建一个shape为[self.bs, self.n_max_boxes, na]的全0的bbox_scores, 将pd_scores的预测分类分数赋值到对应的bbox_scores中(只赋值mask_gt中为1的位置)  相当于公式中的s
     
     得到pd_boxes, pd_boxes是[b, n_max_boxes, na, 4][mask_gt] = [N, 4], (原始的pd_bboxes是[b, na, 4], expand之后就是[b, n_max_boxes, na, 4], 这个可以解释成每一个gt对应[b, na, 4])

     得到gt_boxes, gt_bboxes是[b, n_max_boxes, na, 4][mask_gt] = [N, 4], (原始的gt_bboxess是[b, n_max_boxes, 4], expand之后就是[b, n_max_boxes, na, 4], 这个可以解释为每一个grid对应一个[b, n_max_boxes, 4])

     得到overlaps(shape(b, n_max_boxes, na)), 相当于公式中的ciou

     计算匹配度
  3. 对一个预测框与多个真实框匹配测情况进行处理，保留ciou值最大的真实框。

* 代码解读
```python
先验知识
shape(bs, n_max_labels, h*w)
n_max_labels: 一个batch中一张图片中的gt的数量(一个batch中所有图片的gt的数量进行比较, 选出gt数量最大的那个作为n_max_labels)
h*w = 80*80 + 40*40 + 20 * 20: 既是锚点的数量也是预测框的数量
```
  
```python
class TaskAlignedAssigner(nn.Module):
    """
    A task-aligned assigner for object detection.

    This class assigns ground-truth (gt) objects to anchors based on the task-aligned metric,
    which combines both classification and localization information.

    Attributes:
        topk (int): The number of top candidates to consider.
        num_classes (int): The number of object classes.
        alpha (float): The alpha parameter for the classification component of the task-aligned metric.
        beta (float): The beta parameter for the localization component of the task-aligned metric.
        eps (float): A small value to prevent division by zero.
    """

    def __init__(self, topk=13, num_classes=80, alpha=1.0, beta=6.0, eps=1e-9):
        """Initialize a TaskAlignedAssigner object with customizable hyperparameters."""
        super().__init__()
        self.topk = topk  # 每个gt box最多选择topk个候选框作为正样本
        self.num_classes = num_classes
        self.bg_idx = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """
        Compute the task-aligned assignment.
        Reference https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py

        Args:
            pd_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            pd_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            anc_points (Tensor): shape(num_total_anchors, 2)  这里的anc_points已经是映射到原始图片上的坐标中心点了
            gt_labels (Tensor): shape(bs, n_max_boxes, 1)
            gt_bboxes (Tensor): shape(bs, n_max_boxes, 4)
            mask_gt (Tensor): shape(bs, n_max_boxes, 1)

        Returns:
            target_labels (Tensor): shape(bs, num_total_anchors)
            target_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            target_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            fg_mask (Tensor): shape(bs, num_total_anchors)
            target_gt_idx (Tensor): shape(bs, num_total_anchors)
        """
        self.bs = pd_scores.size(0)
        self.n_max_boxes = gt_bboxes.size(1)

        # 如果不存在真实框, 直接返回结果
        if self.n_max_boxes == 0:
            device = gt_bboxes.device
            return (torch.full_like(pd_scores[..., 0], self.bg_idx).to(device), torch.zeros_like(pd_bboxes).to(device),
                    torch.zeros_like(pd_scores).to(device), torch.zeros_like(pd_scores[..., 0]).to(device),
                    torch.zeros_like(pd_scores[..., 0]).to(device))
        
        # 真实框的mask，正负样本的匹配程度，正负样本的IoU值
        mask_pos, align_metric, overlaps = self.get_pos_mask(pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points,
                                                             mask_gt)

        # 对一个正样本匹配多个真实框的情况进行调整
        # target_gt_idx(shape(bs, h*w)): [b][0]=1表示的是索引为0的pd对应gt(n_max_boxes) 索引为1的, 隐含了gt与pd的索引信息
        # fg_mask(shape(bs, h*w)): fg_mask代表的是有哪些锚点为1, 也就是有哪些锚点是正样本
        # mask_pos(shape(bs, n_max_boxes, h*w)): 值为1的就是第j个pd是第i个gt的正样本, 值为0的就是第j个pd是第i个gt的负样本, i在0~(n_max_boxes-1)之间, j在0~(h*w-1)之间
        target_gt_idx, fg_mask, mask_pos = select_highest_overlaps(mask_pos, overlaps, self.n_max_boxes)

        # Assigned target
        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)

        # Normalize
        # 见参考链接知乎链接最后一部分
        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(axis=-1, keepdim=True)  # b, max_num_obj
        pos_overlaps = (overlaps * mask_pos).amax(axis=-1, keepdim=True)  # b, max_num_obj
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        """Get in_gts mask, (b, max_num_obj, h*w)."""
        # 筛选锚点在真实框内的真实框  (Tensor): shape(b, n_boxes, h*w)
        mask_in_gts = select_candidates_in_gts(anc_points, gt_bboxes)
        # Get anchor_align metric, (b, max_num_obj, h*w)
        # 预测框和真实框的匹配程度、预测框和真实框的IoU值
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt)
        # Get topk_metric mask, (b, max_num_obj, h*w)
        mask_topk = self.select_topk_candidates(align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool())
        # Merge all mask to a final mask, (b, max_num_obj, h*w)
        # 选择有效真实框, 锚点落在真实框内部, 该锚点对应的预测框与真实框的匹配度是topk
        mask_pos = mask_topk * mask_in_gts * mask_gt

        return mask_pos, align_metric, overlaps

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        """
        Args:
            pd_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            pd_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            gt_labels (Tensor): shape(bs, n_max_boxes, 1)
            gt_bboxes (Tensor): shape(bs, n_max_boxes, 4)
            mask_gt (Tensor): shape(bs, n_max_boxes, na)
        return:
            align_metric (Tensor): shape(bs, max_num_obj, na)  
            返回匹配度, max_num_obj可以理解为gt, na可以理解为pd, 也就是将gt中的每一个都与na中的进行计算匹配度
            overlaps (Tensor): shape(bs, max_num_obj, na)  返回计算公式中的ciou
        """
        """Compute alignment metric given predicted and ground truth bounding boxes."""
        na = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()  # b, max_num_obj, h*w
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        bbox_scores = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)
        
        # ind[0]的值为[[0,...,0], ..., [b, ..., b]]  shape(b, max_num_obj)
        # ind[1]的值为gt_labels  shape(b, max_num_obj)
        # 构建一个shape为[self.bs, self.n_max_boxes, na]的全0的bbox_scores, 
        # pd_scores  shape(b, na, 2) -> pd_scores[ind[0], :, ind[1]]: shape(b, max_num_obj, na)
        # pd_scores[ind[0], :, ind[1]]进行广播机制 ind[0]中的[0, 0], ind[1]中的[0, 0] 得到pd_scores[0, :, 0] 以此进行广播
        # 将pd_scores的预测分类分数赋值到对应的bbox_scores中(只赋值mask_in_gt中为1的位置)
        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)  # 2, b, max_num_obj
        ind[0] = torch.arange(end=self.bs).view(-1, 1).expand(-1, self.n_max_boxes)  # b, max_num_obj
        ind[1] = gt_labels.squeeze(-1)  # b, max_num_obj
        # Get the scores of each grid for each gt cls
        # 这里的bbox_scores就是TaskAlignedAssigner中计算公式中的s
        bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]  # b, max_num_obj, h*w  

        # (b, max_num_obj, 1, 4), (b, 1, h*w, 4)
        # pd_boxes shape(N, 4) N是mask_gt中为True的总数量
        # pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1): shape(b, max_num_obj, na, 4)  mask_gt: shape[b, num_max_obj, na]
        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]
        # 这里的overlaps就是TaskAlignedAssigner中计算公式中的ciou
        overlaps[mask_gt] = bbox_iou(gt_boxes, pd_boxes, xywh=False, CIoU=True).squeeze(-1).clamp_(0)

        # 计算匹配程度
        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps
    

    '''
    本人理解的
    metrics是匹配度(shape(b, max_num_obj, h*w)), 在最后一维度选取前self.topk个最大值, 得到前10个匹配度最高的最后一维度的索引值, 也就是topk_idxs的值在0-8399之间
    如果真实框是无效的, 将与之匹配的topk_idxs正样本索引值置为 0
    将topk_idxs中的索引以scatter_add_的方式映射回count_tensor(shape(b, max_num_obj, h*w))
    映射方式可参考链接https://blog.csdn.net/qq_33866063/article/details/120754829
    
    映射方式：
    这里的max_num_obj可以理解为gt, h*w可以理解为pred  
    举个例子[32, 22, 10]中32表示batch_size, 22表示max_num_obj, 10表示h*w
    在一次 for k in range(self.topk):中以[32, 22, 1]为例 
    如topk_idxs[31][21][0]的值是8300,也就是说第32张图片中的第22个gt与第8300的pd匹配度是位于前10中, 也就是count_tensor[31][21][8300]=1
 
    count_tensor(shape(b, max_num_obj, h*w)):
    count_tensor[31][21][8300]=1表示第32张图片第22个gt对应的是第8300个pd, 这个pd是正样本
    其中count_tensor中为1表示是正样本, 为0表示是负样本

    只有当出现补零的gt_boxes时, 才会出现count_tensor > 1的情况一般来说, 因此才使用count_tensor.masked_fill_(count_tensor > 1, 0)将值置为0
    count_tensor>1的条件是topk_idxs(shape(32, 22, 10))中最后一维10中存在相同的两个数, 而出现补零gt_box时, 就topk_idxs.masked_fill_(~topk_mask, 0), 将最后一维10进行置0操作
    '''
    def select_topk_candidates(self, metrics, largest=True, topk_mask=None):
        """
        Select the top-k candidates based on the given metrics.

        Args:
            metrics (Tensor): A tensor of shape (b, max_num_obj, h*w), where b is the batch size,
                              max_num_obj is the maximum number of objects, and h*w represents the
                              total number of anchor points.
            largest (bool): If True, select the largest values; otherwise, select the smallest values.
            topk_mask (Tensor): An optional boolean tensor of shape (b, max_num_obj, topk), where
                                topk is the number of top candidates to consider. If not provided,
                                the top-k values are automatically computed based on the given metrics.

        Returns:
            (Tensor): A tensor of shape (b, max_num_obj, h*w) containing the selected top-k candidates.
        """

        # (b, max_num_obj, topk)
        # # 第一个值为排序的数组，第二个值为该数组中获取到的元素在原数组中的位置索引
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=largest)
        # 如果没有给出有效真实框的mask，通过真实框和预测框的匹配程度确定真实框的有效性
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True)[0] > self.eps).expand_as(topk_idxs)
        # (b, max_num_obj, topk)
        # 如果真实框是无效的，将与之匹配的正样本索引值置为 0  
        topk_idxs.masked_fill_(~topk_mask, 0)

        # (b, max_num_obj, topk, h*w) -> (b, max_num_obj, h*w)
        count_tensor = torch.zeros(metrics.shape, dtype=torch.int8, device=topk_idxs.device)
        ones = torch.ones_like(topk_idxs[:, :, :1], dtype=torch.int8, device=topk_idxs.device)
        for k in range(self.topk):
            # Expand topk_idxs for each value of k and add 1 at the specified positions
            count_tensor.scatter_add_(-1, topk_idxs[:, :, k:k + 1], ones)
        # count_tensor.scatter_add_(-1, topk_idxs, torch.ones_like(topk_idxs, dtype=torch.int8, device=topk_idxs.device))
        # filter invalid bboxes
        count_tensor.masked_fill_(count_tensor > 1, 0)

        return count_tensor.to(metrics.dtype)

    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        """
        Compute target labels, target bounding boxes, and target scores for the positive anchor points.

        Args:
            gt_labels (Tensor): Ground truth labels of shape (b, max_num_obj, 1), where b is the
                                batch size and max_num_obj is the maximum number of objects.
            gt_bboxes (Tensor): Ground truth bounding boxes of shape (b, max_num_obj, 4).
            target_gt_idx (Tensor): Indices of the assigned ground truth objects for positive
                                    anchor points, with shape (b, h*w), where h*w is the total
                                    number of anchor points.
            fg_mask (Tensor): A boolean tensor of shape (b, h*w) indicating the positive
                              (foreground) anchor points.

        Returns:
            (Tuple[Tensor, Tensor, Tensor]): A tuple containing the following tensors:
                - target_labels (Tensor): Shape (b, h*w), containing the target labels for
                                          positive anchor points.
                - target_bboxes (Tensor): Shape (b, h*w, 4), containing the target bounding boxes
                                          for positive anchor points.
                - target_scores (Tensor): Shape (b, h*w, num_classes), containing the target scores
                                          for positive anchor points, where num_classes is the number
                                          of object classes.
        """

        # Assigned target labels, (b, 1)
        # 这三行是一体的, 因为gt_labels被展开了, bs*n_max_boxes
        # 所以要进行第二行代, 由于batch_ind是0~(bs-1)之间, target_gt_idx在0~(n_max_boxes-1), 因此处理后的代码target_gt_idx是在0~(n_max_boxes-1 + (bs-1)*n_max_boxes)之间
        # 第三行代码是一种广播机制, 假设target_gt_idx[1][20]=30(30这个值一定在(1*n_max_boxes)~(1*n_max_boxes+n_max_boxes-1))
        # 也就是target_labels[1][20]=gt_labels[30], target_labels中的值相当于在第一张图片第20个锚点处对应的是第一张图片第(30-n_max_boxes)的label值
        # 假设target_gt_idx[0][1] = 0, 这个0是mask_pos[0, :, 1]中的最大值为0, 也就代表pd1这一个anchor并没有匹配到gt,是负样本, 但是gt_labels[0]确是第一张图片的第一个gt_box的label值, 所以在下方需要将target_score中的负样本进行过滤(置0)
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes  # (b, h*w)
        target_labels = gt_labels.long().flatten()[target_gt_idx]  # (b, h*w)

        # Assigned target boxes, (b, max_num_obj, 4) -> (b, h*w)
        target_bboxes = gt_bboxes.view(-1, 4)[target_gt_idx]

        # Assigned target scores
        target_labels.clamp_(0)

        # 10x faster than F.one_hot()
        target_scores = torch.zeros((target_labels.shape[0], target_labels.shape[1], self.num_classes),
                                    dtype=torch.int64,
                                    device=target_labels.device)  # (b, h*w, 80)
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)

        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes)  # (b, h*w, 80)
        # 过滤负样本, 负样本的位置的target_scores都为0, 只保留正样本的
        # target_bboxes的在生成box损失的会过滤
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)
		
        return target_labels, target_bboxes, target_scores

```
对一个预测框与多个真实框匹配测情况进行处理，保留ciou值最大的真实框, 虽然一个pd不能对应多个gt, 但是一个gt可以对应多个pd。函数调用如下
```python
def select_highest_overlaps(mask_pos, overlaps, n_max_boxes):
    """if an anchor box is assigned to multiple gts,
        the one with the highest iou will be selected.

    Args:
        mask_pos (Tensor): shape(b, n_max_boxes, h*w)
        overlaps (Tensor): shape(b, n_max_boxes, h*w)
    Return:
        target_gt_idx (Tensor): shape(b, h*w)
        fg_mask (Tensor): shape(b, h*w)
        mask_pos (Tensor): shape(b, n_max_boxes, h*w)
    """
    # (b, n_max_boxes, h*w) -> (b, h*w)
    # 预测输出总共会有h*w个预测框, n_max_boxes对应的是gt, 如果这一维度存在sum求和大于1的情况
    # h*w=8400, 假设[b][0] > 1, 也就是[0]处的预测框同时被分给多个gt 
    fg_mask = mask_pos.sum(-2)
    if fg_mask.max() > 1:  # one anchor is assigned to multiple gt_bboxes
        # fg_mask.unsqueeze(1) > 1是将fg_mask变为bool值
        mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(-1, n_max_boxes, -1)  # (b, n_max_boxes, h*w)
        # overlaps就是CIoU  选择gt与pd ciou最大的那个位置索引  这个索引的值的维度是1, 值也就是在0-n_max_boxes-1之间
        max_overlaps_idx = overlaps.argmax(1)  # (b, h*w)

        # is_max_overlaps: [b, n_max_boxes, h*w], 中将is_max_overlaps中对应的n_max_boxes的维度赋值为1
        # 这个跟select_topk_candidates中的运用有异曲同工之妙
        # 最终的目的就是筛选出gt与pd中CIoU最大的那一维, 将pd对应的多个gt中CIoU最大的那个赋值为1, 其余赋值为0
        is_max_overlaps = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
        is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)
        
        # 用mask_multi_gts中为True的那部分用is_max_overlaps
        # is_max_overlaps是已经是挑选ciou最大值之后的了, 它会覆盖mask_multi_gts中为True, 也就是一个预测框对应多个gt的那部分, ciou最大那一个gt赋值为1, 其余的赋值为0
        # 如果没有一个pd没有对应多个gt, 那么还是直接将原来的值mask_pos赋值给mask_multi_gts
        mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos).float()  # (b, n_max_boxes, h*w)
        fg_mask = mask_pos.sum(-2)
    # Find each grid serve which gt(index)
    target_gt_idx = mask_pos.argmax(-2)  # (b, h*w)
    return target_gt_idx, fg_mask, mask_pos
```

* def select_candidates_in_gts(xy_centers, gt_bboxes, eps=1e-9):介绍
```python
"""
这个函数select_candidates_in_gts的目的是在给定一组中心点(anchor centers)和一组ground truth bounding boxes (gt_bboxes)的情况下,
选择那些与gt_bboxes有重叠的anchor中心, 重叠的意思是anchor的中心点落在了gt_boxes的内部

函数的输入参数如下：
xy_centers(Tensor): 形状为(h*w, 2)的张量, 表示每个anchor box的中心点坐标。每一行包含一个中心点的(x, y, x, y)坐标。
gt_bboxes(Tensor): 形状为(b, n_boxes, 4)的张量, 表示每个样本的n_boxes个ground truth bounding boxes的坐标。每个bounding box由左上角坐标和右下角坐标组成。
"""
def select_candidates_in_gts(xy_centers, gt_bboxes, eps=1e-9):
    """select the positive anchor center in gt

    Args:
        xy_centers (Tensor): shape(h*w, 4) 错误 xy_centers的shape应该是(h*w, 2)
        gt_bboxes (Tensor): shape(b, n_boxes, 4)
    Return:
        (Tensor): shape(b, n_boxes, h*w)
    """
    n_anchors = xy_centers.shape[0]
    bs, n_boxes, _ = gt_bboxes.shape
    # 计算gt_bboxes的左上角坐标(lt)和右下角坐标(rb)。将gt_bboxes重塑为(b*n_boxes, 1, 4), 然后使用chunk(2, 2)将其沿第2维(通道维度)分割成两部分。
    lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)  # left-top, right-bottom
    # 计算每个anchor中心相对于每个ground truth bounding box的偏移量。首先, 将xy_centers添加一个新的维度(维度大小为1)，得到形状为(1, h*w, 4)的张量。
    # 然后, 分别计算anchor中心与每个ground truth bounding box左上角和右下角坐标的差值, 
    # 并将这两个差值连接在一起，得到形状为(bs, n_boxes, n_anchors, 4)的张量bbox_deltas。
    bbox_deltas = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2).view(bs, n_boxes, n_anchors, -1)
    
    # 对于每个anchor中心和每个ground truth bounding box，计算它们之间的最小距离(在x轴和y轴上)
    # 这可以通过对bbox_deltas沿第3维(anchor中心维度)求最小值来实现, 结果是一个形状为(bs, n_boxes, h*w)的张量。
    # 判断这些最小距离是否大于一个很小的阈值eps(默认为1e-9)。如果大于eps，则认为该anchor中心与对应的ground truth bounding box有重叠。
    # 返回一个形状为(bs, n_boxes, h*w)的张量, 其中值为1表示对应的anchor中心与ground truth bounding box有重叠，值为0表示没有重叠。
    # return (bbox_deltas.min(3)[0] > eps).to(gt_bboxes.dtype)
    return bbox_deltas.amin(3).gt_(eps)

```


为更好的解释上面的函数，现举一个例子
```python
# 现假设bs, n_max_boxes, h*w 分别为 1， 3， 4
# n_max_boxes对应着gt, h*w对应着pd
>>>mask_pos = torch.tensor([[[1, 0, 0, 1],
                          [0, 0, 0, 1],
                          [1, 1, 1, 1]]])

# ciou为随机的0-1之间
>>>overlaps = torch.rand((1, 3, 4))
tensor([[[0.0913, 0.3341, 0.2598, 0.5922],
         [0.2369, 0.4138, 0.8834, 0.0176],
         [0.9079, 0.6434, 0.3520, 0.6427]]])

>>>fg_mask = mask_pos.sum(-2)
输出: tensor([[2, 1, 1, 3]]) 
显然这里有大于1的元素, 2=mask_pos[0][0][0] + mask_pos[0][1][0] + mask_pos[0][2][0], 显然在pd位于0的位置对应了gt0和gt2两个gt

mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(-1, 3, -1)  # (b, n_max_boxes, h*w)
tensor([[[ True, False, False,  True],
         [ True, False, False,  True],
         [ True, False, False,  True]]])

>>>max_overlaps_idx = overlaps.argmax(1)  # (b, h*w)
tensor([[2, 2, 1, 2]])
这里挑选出来的是在dim=1上, ciou最大的那个索引, 以第一个2为例: overlaps[0][2][0]=0.9079是overlaps [0, :, 0]中值最大的

>>>is_max_overlaps = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
>>>is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)  # 第一个1是dim, 第二个1是value
tensor([[[0, 0, 0, 0],
         [0, 0, 1, 0],
         [1, 1, 0, 1]]])
max_overlaps_idx在经过unsqueeze后shape为(1, 1, 4)
scatter_是这样使用的, max_overlaps_idx[i][0][j] = x , is_max_overlaps[i][x][j] = 1, 这个1是由scatter_()中最后一个参数决定的
max_overlaps_idx[0][0][0] = 2, 即将is_max_overlaps[0][2][0]=1, max_overlaps_idx[0][0][1]=2, 即将is_max_overlaps[0][2][1], max_overlaps_idx[0][0][2]=1
即将is_max_overlaps[0][1][2]=1 ......

>>>mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos).float()  # (b, n_max_boxes, h*w)
tensor([[[0., 0., 0., 0.],
         [0., 0., 0., 0.],
         [1., 1., 1., 1.]]])
mask_multi_gts中为True的值由is_max_overlaps中相同位置的值代替, mask_multi_gts中为False的值由mask_pos中相同位置的值代替

>>>fg_mask = mask_pos.sum(-2)
tensor([[1., 1., 1., 1.]])
fg_mask代表的是有哪些锚点也就是预测框为1

>>>target_gt_idx = mask_pos.argmax(-2)  # (b, h*w)
tensor([[2, 2, 2, 2]])
target_gt_idx: [b][0]=2表示的是索引为0的pd对应gt(n_max_boxes) 索引为2的, 隐含gt与pd的索引信息
```
由上面的例子可知, 假如一个预测对应多个gt, 只将CIoU最大的位置保留下来

#### 解释一张图片不满足n_max_boxes个gt时, 补零操作后, 怎么消除这些影响
一个预测框只对应一个gt, 但是一个gt可以对应多个pd
mask定义为补零的gt_boxes

1. 首先get_targets函数中的target_labels = gt_labels.long().flatten()[target_gt_idx], 这个将gt_labels[bs, n_max_boxes, 1]转化为targest_labels[bs, h*w]。如果target_gt_idx中的值没有出现补零的gt_boxes的索引, 那么在调用gt_labels中的值是就相当于去掉了补零的gt_boxes,消除了补零的gt_boxes的影响
2. target_gt_idx(shape(bs, h*w)), 只要target_gt_idx的值中没有mask对应的索引值即可, 在经过target_labels = gt_labels.long().flatten()[target_gt_idx]后就会直接过滤掉mask
3. target_gt_idx来自于select_highest_overlaps函数, target_gt_idx = mask_pos.argmax(-2), 这个mask_pos(shape(bs, n_max_boxes, h*w))就是最终的gt与pd的分配情况(细看可以看上面的注释)。
```python
pd0意思是h*w中的第一个锚点对应的pd, 这个0是下标
gt0意思是第一个gt

假设gt0、gt1是真实boxes, gt2是补零的boxes
当mask_pos为(bs=1, n_max_boxes=3, pd=3)
torch.tensor([[[1, 0, 0],
			   [0, 0, 1], 
			   [0, 0, 0]]])
这个gt2一定是全0的(来源select_topk_candidates)
正样本就只有两个, 分别是pd0, pd2, pd0分配给gt0, pd2也分配给gt1

target_gt_idx
torch.tensor([[0, 0, 1]])
0就表示gt0, 可以看到pd1也分配给了gt0, 这样就可以看到补零的gt2被过滤掉了, 补零的gt_boxes的索引一定不会出现在target_gt_idx中
```

### 参考链接
[https://zhuanlan.zhihu.com/p/633094573](https://zhuanlan.zhihu.com/p/633094573)
[https://blog.csdn.net/YXD0514/article/details/132116133](https://blog.csdn.net/YXD0514/article/details/132116133)
