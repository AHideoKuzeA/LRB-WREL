import datetime
import os
import time
import torch
import torch.utils.data
import wandb
import cv2
import random
import transforms as T
import utils
import numpy as np
import gc
import operator
from functools import reduce
from bert.modeling_bert import BertModel
from lib import segmentation
from loss.loss import WeakSemiSupervisedLoss, SupervisedLoss
import torch.distributed as dist
import torch.nn as nn
os.environ["WANDB_MODE"]="offline"
from caption_prompt_learner import PromptLearnerForBERT
from torch.utils.data.distributed import DistributedSampler

caption_step_counter = [0]
log_every_caption = 100000000

@torch.no_grad()
def update_ema_variables(model, ema_model, alpha=0.999):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

def get_ema_momentum(epoch: int, step: int, steps_per_epoch: int, final: float = 0.9995) -> float:
    steps_per_epoch = max(1, int(steps_per_epoch))
    progress = epoch + float(step) / steps_per_epoch

    if progress < 1.0:
        return 0.95 + (0.99 - 0.95) * (progress / 1.0)
    elif progress < 3.0:
        return 0.99 + (0.999 - 0.99) * ((progress - 1.0) / 2.0)
    else:
        return final


def seed_everything(seed=2401):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_dataset(image_set, transform, args):
    from data.dataset_train_student import ReferDataset_stu
    ds = ReferDataset_stu(args,
                      split=image_set,
                      image_transforms=transform,
                      target_transforms=None
                      )
    num_classes = 2

    return ds, num_classes


def IoU(pred, gt):
    pred = pred.argmax(1)

    intersection = torch.sum(torch.mul(pred, gt))
    union = torch.sum(torch.add(pred, gt)) - intersection

    if intersection == 0 or union == 0:
        iou = 0
    else:
        iou = float(intersection) / float(union)
    return iou, intersection, union


def get_transform(args):
    transforms = [
                  T.Resize(args.img_size, args.img_size),
                  T.ToTensor(),
                  T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                  ]
    return T.Compose(transforms)


def count_caption_steps_exact(dataset, weak_id_set, batch_size, world_size, rank,
                              epochs, update_every, inner_steps, drop_last=True, seed=0):
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=seed)
    total_steps = 0
    steps_per_epoch = []

    N = len(dataset)
    for e in range(epochs):
        sampler.set_epoch(e)
        indices = list(iter(sampler))
        num_full_batches = len(indices) // batch_size
        if not drop_last and (len(indices) % batch_size != 0):
            num_batches = num_full_batches + 1
        else:
            num_batches = num_full_batches


        weak_batches = 0
        for b in range(num_batches):
            start = b * batch_size
            end   = min((b + 1) * batch_size, len(indices))
            if drop_last and (end - start) < batch_size:
                break
            batch_idx = indices[start:end]
            has_weak = any(int(dataset.ref_ids[i]) in weak_id_set for i in batch_idx)
            if has_weak:
                weak_batches += 1

        trig = weak_batches // max(1, update_every)
        steps_e = trig * max(1, inner_steps)
        steps_per_epoch.append(steps_e)
        total_steps += steps_e

    return total_steps, steps_per_epoch


def evaluate(model, data_loader, bert_model, epoch, cap_sup_criterion):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test: "
    total_its = 0
    acc_ious = 0

    cum_I, cum_U = 0, 0
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    seg_total = 0
    mean_IoU = []
    total_loss = 0

    with torch.no_grad():
        for data in metric_logger.log_every(data_loader, 100, header):
            total_its += 1
            image, target, sentences, attentions, sample_id = data
            pixels = cv2.countNonZero(target.data.numpy()[0]) / 230400.
            image, target, sentences, attentions = image.cuda(non_blocking=True),\
                                                   target.cuda(non_blocking=True),\
                                                   sentences.cuda(non_blocking=True),\
                                                   attentions.cuda(non_blocking=True)

            sentences = sentences.squeeze(1)
            attentions = attentions.squeeze(1)


            if bert_model is not None:
                last_hidden_states = bert_model(sentences, attention_mask=attentions)[0]
                embedding = last_hidden_states.permute(0, 2, 1)  
                attentions = attentions.unsqueeze(dim=-1)  
                output = model(image, embedding, l_mask=attentions)
            else:
                output = model(image, sentences, l_mask=attentions)

            iou, I, U = IoU(output, target)
            loss = cap_sup_criterion(output, target)
            total_loss += loss.item()
            acc_ious += iou
            mean_IoU.append(iou)
            cum_I += I
            cum_U += U
            for n_eval_iou in range(len(eval_seg_iou_list)):
                eval_seg_iou = eval_seg_iou_list[n_eval_iou]
                seg_correct[n_eval_iou] += (iou >= eval_seg_iou)
            seg_total += 1
        iou = acc_ious / total_its

    mean_IoU = np.array(mean_IoU)
    mIoU = np.mean(mean_IoU)
    print('Final results:')
    print('Mean IoU is %.2f\n' % (mIoU * 100.))
    results_str = ''
    for n_eval_iou in range(len(eval_seg_iou_list)):
        results_str += '    precision@%s = %.2f\n' % \
                       (str(eval_seg_iou_list[n_eval_iou]), seg_correct[n_eval_iou] * 100. / seg_total)
    results_str += '    overall IoU = %.2f\n' % (cum_I * 100. / cum_U)
    print(results_str)

    if args.local_rank == 0:
        wandb.log({
            "val mIoU": mIoU,
            "val oiou": cum_I * 100. / cum_U,
            "val Loss": total_loss / total_its})

    return 100 * iou, 100 * cum_I / cum_U


def train_one_epoch(student_model, teacher_model, criterion, cap_sup_criterion, optimizer, caption_optimizer, data_loader, lr_scheduler,
                    lr_scheduler_caption, epoch, print_freq,iterations, bert_model, caption_bank,caption_inner_steps,caption_update_every, start_epoch):
    student_model.train()
    teacher_model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)
    steps_per_epoch = len(data_loader)

    train_loss = 0
    total_its = 0
    train_loss_caption = 0.0
    iterations_caption = 0


    bank_mod = caption_bank.module if hasattr(caption_bank, "module") else caption_bank
    bert_mod = bert_model.module if hasattr(bert_model, "module") else bert_model


    inner_steps = caption_inner_steps       
    update_every = caption_update_every     


    for i, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        total_its += 1
        image, target, sentences, attentions, sample_id = data
        image, target, sentences, attentions, sample_id = image.cuda(non_blocking=True),\
                                               target.cuda(non_blocking=True),\
                                               sentences.cuda(non_blocking=True),\
                                               attentions.cuda(non_blocking=True),\
                                               sample_id.cuda(non_blocking=True)

        sentences = sentences.squeeze(1)
        attentions = attentions.squeeze(1)


        weak_mask = torch.tensor(
            [int(sid) in bank_mod.sample_id_to_index for sid in sample_id],
            dtype=torch.bool, device=sample_id.device
        )



        if weak_mask.any() and (i % update_every == 0) :
            image_w   = image[weak_mask]
            target_w  = target[weak_mask]
            sid_w     = sample_id[weak_mask]
            sent_w    = sentences[weak_mask]
            mask_w    = attentions[weak_mask]  


            with torch.no_grad():
                base_embeds_w = bert_mod.embeddings.word_embeddings(sent_w.long()) 

            for _ in range(inner_steps):

                filled_embeds_w, filled_mask_w = bank_mod(sid_w, base_embeds_w, mask_w) 
                bert_out_w = bert_mod(inputs_embeds=filled_embeds_w, attention_mask=filled_mask_w)[0]
                bert_embed_w = bert_out_w.permute(0, 2, 1)   
                l_mask_w = filled_mask_w.unsqueeze(-1)       

                pseudo_logits = teacher_model(image_w, bert_embed_w, l_mask=l_mask_w)

                caption_optimizer.zero_grad()
                loss_caption = cap_sup_criterion(pseudo_logits, target_w)
                loss_caption.backward()
                caption_optimizer.step()
                lr_scheduler_caption.step()

                train_loss_caption += loss_caption.item()
                iterations_caption += 1
                caption_step_counter[0] += 1

                if utils.is_main_process() and (caption_step_counter[0] % log_every_caption == 0):
                    cur_lr_cap = caption_optimizer.param_groups[0]["lr"]
                    print(f"[cap-step {caption_step_counter[0]:6d}] "
                        f"epoch={epoch} it={i} inner={iterations_caption} "
                        f"loss_cap={loss_caption.item():.6f} lr_cap={cur_lr_cap:.6e}")
                    wandb.log({
                        "cap/loss": loss_caption.item(),
                        "cap/lr":   cur_lr_cap,
                        "cap/epoch": epoch,
                        "cap/global_step": caption_step_counter[0],
            })


            del image_w, target_w, sid_w, sent_w, mask_w, base_embeds_w, filled_embeds_w, filled_mask_w
            del bert_out_w, bert_embed_w, l_mask_w, pseudo_logits, loss_caption



        input_embeds = bert_model.module.embeddings.word_embeddings(sentences)
        attn_mask = attentions
        inputs_embeds_aug = input_embeds
        attn_mask_aug = attn_mask
        if weak_mask.any():
            sid_sub = sample_id[weak_mask]
            emb_sub = input_embeds[weak_mask]
            m_sub = attn_mask[weak_mask]

            filled_embeds_sub, filled_mask_sub = caption_bank(sid_sub, emb_sub, m_sub)
            filled_embeds_sub = filled_embeds_sub.detach()
            filled_mask_sub   = filled_mask_sub.detach()


            inputs_embeds_aug = input_embeds.clone()
            attn_mask_aug = attn_mask.clone()
            inputs_embeds_aug[weak_mask] = filled_embeds_sub
            attn_mask_aug[weak_mask] = filled_mask_sub


        last_hidden_states = bert_model(
            inputs_embeds=inputs_embeds_aug,    
            attention_mask=attn_mask_aug      
        )[0]


        bert_embedding = last_hidden_states.permute(0, 2, 1) 
        l_mask = attn_mask_aug.unsqueeze(-1)                 


        output = student_model(image, bert_embedding, l_mask=l_mask)
        student_logits_sup = output
        target_sup = target

        student_logits_weak = None
        teacher_logits_weak = None

        total_loss, loss_items = criterion(
            student_logits_sup=student_logits_sup,
            target_sup=target_sup,
            epoch=epoch
        )

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        lr_scheduler.step()

        cur_alpha = get_ema_momentum(epoch, i, steps_per_epoch, final=0.9995)
        update_ema_variables(student_model, teacher_model, alpha=cur_alpha)

        torch.cuda.synchronize()
        train_loss += total_loss.item()
        iterations += 1
        metric_logger.update(loss=total_loss.item(), lr=optimizer.param_groups[0]["lr"])

        del image, target, sentences, attentions, total_loss, output, data
        if bert_model is not None:
            del last_hidden_states, bert_embedding

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    if args.local_rank == 0:
        wandb.log({
            "Train Loss": train_loss / total_its,})

def init_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        print(f"RANK and WORLD_SIZE in environment: {rank}/{world_size}")
    else:
        print('Not using distributed mode')
        return False, 0, 1, 0


    torch.cuda.set_device(local_rank)
    

    dist_backend = 'nccl'
    dist_url = 'env://'
    
    print('| distributed init (rank {}): {}'.format(rank, dist_url), flush=True)
    
    try:
        if not dist.is_initialized():
            dist.init_process_group(
                backend=dist_backend, 
                init_method=dist_url,
                world_size=world_size, 
                rank=rank
            )
            dist.barrier()
        else:
            print("Process group already initialized, skipping initialization.")
    except Exception as e:
        print(f"Failed to initialize distributed training: {e}")
        print("Falling back to single GPU training")
        return False, 0, 1, 0
    
    return True, rank, world_size, local_rank

def main(args):

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    try:
        distributed, rank, world_size, local_rank = init_distributed()
        args.distributed = distributed
        args.rank = rank
        args.world_size = world_size
        args.gpu = local_rank
        args.local_rank = local_rank
        
        if args.distributed:
            print(f"Running in distributed mode with {world_size} GPUs, rank: {rank}, local_rank: {local_rank}")
            args.batch_size = int(args.batch_size / args.world_size)
            args.workers = int((args.workers + args.world_size - 1) / args.world_size)
    except Exception as e:
        print(f"Error in distributed initialization: {e}")
        print("Falling back to single GPU training")
        args.distributed = False
        args.rank = 0
        args.world_size = 1
        args.gpu = 0
        args.local_rank = 0
        
    dataset, num_classes = get_dataset("train",
                                       get_transform(args=args),
                                       args=args)

    sample_id_list = dataset.ref_ids
    dataset_test, _ = get_dataset("val",
                                  get_transform(args=args),
                                  args=args)


    saved_bank = torch.load(args.caption_embedding_path)
    saved_embeddings = saved_bank["embedding"] 
    saved_id2idx = saved_bank["sample_id_to_index"] 
    
    sample_id_list_weak = list(saved_id2idx.keys())
    n_prompt = saved_embeddings.shape[1]
    embedding_dim = saved_embeddings.shape[2]

    caption_bank = PromptLearnerForBERT(
        sample_id_list=sample_id_list_weak,
        embedding_dim=embedding_dim,
        n_prompt=n_prompt
    )

    device = torch.device("cuda", args.local_rank)
    caption_bank = caption_bank.to(device)


    pe = caption_bank.prompt_embeddings 
    for sid, idx in caption_bank.sample_id_to_index.items():
        saved_idx = saved_id2idx[sid]
        pe.data[idx].copy_(saved_embeddings[saved_idx].to(pe.device))



    print(f"local rank {args.local_rank} / global rank {utils.get_rank()} successfully built train dataset.")
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank,
                                                                    shuffle=True)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)


    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers, pin_memory=args.pin_mem, drop_last=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, sampler=test_sampler, num_workers=args.workers)
    

    weak_id_set = set(saved_id2idx.keys())
    total_cap_steps, cap_steps_each_epoch = count_caption_steps_exact(
        dataset=dataset,
        weak_id_set=weak_id_set,
        batch_size=args.batch_size,
        world_size=utils.get_world_size(),
        rank=utils.get_rank(),
        epochs=args.epochs,
        update_every=args.caption_update_every,
        inner_steps=args.caption_inner_steps,
        drop_last=True,
        seed=0 
    )

    if utils.is_main_process():
        print(f"[Caption LR] total caption steps (exact) = {total_cap_steps}")
        print(f"[Caption LR] steps per epoch (exact) = {cap_steps_each_epoch}")


    print(args.model)
    student_model = segmentation.__dict__[args.model](pretrained=args.pretrained_swin_weights, args=args)
    teacher_model = segmentation.__dict__[args.model](pretrained=args.pretrained_swin_weights, args=args)


    student_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(student_model).cuda()
    teacher_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(teacher_model).cuda()

    student_model = torch.nn.parallel.DistributedDataParallel(student_model, device_ids=[args.local_rank], find_unused_parameters=True)
    

    student_model.train()
    teacher_model.eval()

    single_student = student_model.module
    single_teacher = teacher_model


    if args.model != 'lavt_one':
        model_class = BertModel
        bert_model = model_class.from_pretrained(args.ck_bert)
        bert_model.pooler = None  
        bert_model.cuda()
        #bert_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(bert_model)
        bert_model = torch.nn.parallel.DistributedDataParallel(bert_model, device_ids=[args.local_rank], find_unused_parameters=True)
        single_bert_model = bert_model
    else:
        bert_model = None
        single_bert_model = None


    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        single_student.load_state_dict(checkpoint['model'], strict=False)
        single_teacher.load_state_dict(checkpoint['model'], strict=False)
        if args.model != 'lavt_one':
            single_bert_model.load_state_dict(checkpoint['bert_model'])


    for p in teacher_model.parameters():
        p.requires_grad = False


    backbone_no_decay = list()
    backbone_decay = list()
    for name, m in single_student.backbone.named_parameters():
        if 'norm' in name or 'absolute_pos_embed' in name or 'relative_position_bias_table' in name:
            backbone_no_decay.append(m)
        else:
            backbone_decay.append(m)

    if args.model != 'lavt_one':
        params_to_optimize = [
            {'params': backbone_no_decay, 'weight_decay': 0.0},
            {'params': backbone_decay},
            {"params": [p for p in single_student.classifier.parameters() if p.requires_grad]},
            {"params": reduce(operator.concat,
                              [[p for p in single_bert_model.module.encoder.layer[i].parameters()
                                if p.requires_grad] for i in range(10)])},
        ]
    else:
        params_to_optimize = [
            {'params': backbone_no_decay, 'weight_decay': 0.0},
            {'params': backbone_decay},
            {"params": [p for p in single_student.classifier.parameters() if p.requires_grad]},
            {"params": reduce(operator.concat,
                              [[p for p in single_student.text_encoder.encoder.layer[i].parameters()
                                if p.requires_grad] for i in range(10)])},
        ]


    optimizer = torch.optim.AdamW(params_to_optimize,
                                  lr=args.lr,
                                  weight_decay=args.weight_decay,
                                  amsgrad=args.amsgrad
                                  )
    
    optimizer_caption = torch.optim.AdamW(caption_bank.parameters(),
                                          lr=args.caption_lr,
                                           weight_decay=args.weight_decay,
                                           amsgrad=args.amsgrad
                                           )


    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                     lambda x: (1 - x / (len(data_loader) * args.epochs)) ** 0.9)

    
    if total_cap_steps > 0:
        lr_scheduler_caption = torch.optim.lr_scheduler.LambdaLR(
            optimizer_caption,
            lr_lambda=lambda step: (1.0 - step / float(total_cap_steps)) ** 0.9
        )
    else:
        lr_scheduler_caption = torch.optim.lr_scheduler.LambdaLR(
            optimizer_caption, lr_lambda=lambda step: 1.0
        )


    start_time = time.time()
    iterations = 0
    best_oIoU = -0.1
    best_mIoU = -0.1

    resume_epoch = -999


    if args.local_rank == 0:
        wandb.watch(student_model, log="all")
    

    crit_train = WeakSemiSupervisedLoss(
        sup_weight_dice=0.1,          
        unsup_weight=0,             
        thr_init=0.6, thr_final=0.8, 
        epochs_ramp=5,
        temperature_teacher=0.7, temperature_student=1.0
    )

    cap_sup_criterion = SupervisedLoss(weight=0.1)


    for epoch in range(max(0, resume_epoch+1), args.epochs):
        data_loader.sampler.set_epoch(epoch)
        train_one_epoch(student_model, teacher_model, crit_train,cap_sup_criterion,
                        optimizer,optimizer_caption,data_loader,
                        lr_scheduler,lr_scheduler_caption, epoch, args.print_freq,
                        iterations, bert_model, caption_bank,args.caption_inner_steps,args.caption_update_every, args.start_epoch)
        iou, overallIoU = evaluate(student_model, data_loader_test, bert_model, epoch, cap_sup_criterion)
        print('Average object IoU {}'.format(iou))
        print('Overall IoU {}'.format(overallIoU))
        best_o = (best_oIoU < overallIoU)
        best_m = (best_mIoU < iou)
        if single_bert_model is not None:
            dict_to_save = {'student_model': single_student.state_dict(),
                            'bert_model': single_bert_model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'epoch': epoch,
                            'args': args,
                            'lr_scheduler': lr_scheduler.state_dict()}
        else:
            dict_to_save = { 'student_model': single_student.state_dict(),
                            'teacher_model': single_teacher.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'epoch': epoch,
                            'args': args,
                            'lr_scheduler': lr_scheduler.state_dict()}

        if best_o:
            print('Better oIoU epoch: {}\n'.format(epoch))
            utils.save_on_master(dict_to_save, os.path.join(args.output_dir,
                                                            'model_best_oIoU_{}.pth'.format(args.model_id)))
            best_oIoU = overallIoU
            if args.local_rank == 0:
                embed_path = os.path.join(args.output_dir, f"caption_embeddings_{args.model_id}_oIoU_best.pt")
                torch.save({
                    "embedding": caption_bank.module.prompt_embeddings.detach().cpu()
                                if isinstance(caption_bank, torch.nn.parallel.DistributedDataParallel)
                                else caption_bank.prompt_embeddings.detach().cpu(),
                    "sample_id_to_index": caption_bank.module.sample_id_to_index
                                        if isinstance(caption_bank, torch.nn.parallel.DistributedDataParallel)
                                        else caption_bank.sample_id_to_index
                }, embed_path)
                print(f"=> Saved caption embedding to {embed_path}")
        if best_m:
            print('Better mIoU epoch: {}\n'.format(epoch))
            utils.save_on_master(dict_to_save, os.path.join(args.output_dir,
                                                            'model_best_mIoU_{}.pth'.format(args.model_id)))
            best_mIoU = iou
            if args.local_rank == 0:
                embed_path = os.path.join(args.output_dir, f"caption_embeddings_{args.model_id}_oIoU_best.pt")
                torch.save({
                    "embedding": caption_bank.module.prompt_embeddings.detach().cpu()
                                if isinstance(caption_bank, torch.nn.parallel.DistributedDataParallel)
                                else caption_bank.prompt_embeddings.detach().cpu(),
                    "sample_id_to_index": caption_bank.module.sample_id_to_index
                                        if isinstance(caption_bank, torch.nn.parallel.DistributedDataParallel)
                                        else caption_bank.sample_id_to_index
                }, embed_path)
                print(f"=> Saved caption embedding to {embed_path}")
        utils.save_on_master(dict_to_save, os.path.join(args.output_dir,
                                                        'model_last_{}.pth'.format(args.model_id)))
        if args.local_rank == 0:
            wandb.save('model.h5')


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    from args import get_parser
    seed_everything()
    parser = get_parser()
    args = parser.parse_args()
    if args.local_rank == 0:
        wandb.init(project="rmsin_2080")
    utils.init_distributed_mode(args)
    print('Image size: {}'.format(str(args.img_size)))
    main(args)
