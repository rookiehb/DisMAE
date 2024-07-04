import pandas as pd 
from pathlib import Path
import torch
import numpy as np
import util.misc as misc
from torchvision.utils import save_image

def train_one_epoch(model, train_loaders, optimizer, epoch, args, total_len, logger):

    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    total_loss = 0; total = 0; test_acc = 0
    for bid, batch_data in enumerate(train_loaders):
        if bid == total_len:
            break

        if args.sample_pos:
            images = torch.cat([x for x, y, x_pos in batch_data])
            labels = torch.cat([y for x, y, x_pos in batch_data])
            images_pos = torch.cat([x_pos for x, y, x_pos in batch_data]) # positive sample
        else:
            images = torch.cat([x for (ori_x, x), y in batch_data])
            labels = torch.cat([y for _, y in batch_data])
            ori_images = torch.cat([ori_x for (ori_x, x), y in batch_data])
        
        images = images.cuda()
        labels = labels.cuda()

        predict = model(images)
        loss = criterion(predict, labels)
        
        total_loss += loss.item()
        total += labels.size(0)
        test_acc += (predict.argmax(dim=1) == labels).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if bid%(total_len//4)==0 and bid!=0:
            logger.info("Bid:[{}/{}], Loss:{:.4f}, Acc:{:.2f}%".format(bid, total_len, total_loss/bid, test_acc/total*100))


def eval_model(model, eval_iterator, device, epoch, args, total_len, logger, env_samples):

    model.eval()
    infos = "EVAL EPOCH:{} | ".format(epoch)
    accs = []
    with torch.no_grad():
        for name, loader, _ in eval_iterator:
            total = 0; test_acc = 0
            for images, labels in loader:
                
                images = images[0].cuda()
                labels = labels.cuda()
        
                with torch.cuda.amp.autocast():
                    predict = model(images)

                total += labels.size(0)
                test_acc += (predict.argmax(dim=1) == labels).sum().item()
        
            Total_acc = 100*test_acc/total
            infos += "{}: {:.2f}% | ".format(name, Total_acc)
            accs.append(Total_acc)

    accs = np.array(accs)
    env_samples = np.array(env_samples)
    accs = accs[accs.shape[0]//2:] # out
    test_envs = np.array(args.test_envs)

    results = (np.sum(accs)-np.sum(accs[test_envs]))/(accs.shape[0]-len(test_envs))    
    infos += " Avg acc of val set:{:.2f}% | ".format(results)
    
    overall_total_num=0; overall_acc_num=0; avg_acc=0
    for i in range(len(env_samples)):
        if i in args.test_envs:
            overall_acc_num += env_samples[i]*accs[i]
            overall_total_num += env_samples[i]
            avg_acc += accs[i]
            
    infos += "Avg acc of test set:{:.2f}%, Overall:{:.2f}%".format(
            avg_acc/len(args.test_envs), overall_acc_num/overall_total_num)
    
    logger.info("{}".format(infos))
    model.train()
    return results