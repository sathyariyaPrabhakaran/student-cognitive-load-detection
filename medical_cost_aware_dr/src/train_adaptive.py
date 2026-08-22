from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader

from data import RetinaDataset, discover, stratified_split
from models import build_lightweight, build_expert
from adaptive_router import AdaptiveRouter


def train_epoch(model, loader, optimizer, device):
    model.train(); loss_fn = nn.CrossEntropyLoss(); total = 0.0
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(); out = model(x); loss = loss_fn(out, y); loss.backward(); optimizer.step(); total += float(loss.item()) * len(y)
    return total / len(loader.dataset)


def predict(model, loader, device):
    model.eval(); probs=[]; labels=[]; paths=[]; elapsed=0.0
    with torch.no_grad():
        for x, y, p in loader:
            x=x.to(device); start=time.perf_counter(); out=model(x); elapsed += time.perf_counter()-start
            probs.append(torch.softmax(out, dim=1).cpu().numpy()); labels.extend(y.numpy()); paths.extend(p)
    return np.vstack(probs), np.asarray(labels), paths, elapsed


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-dir', required=True); ap.add_argument('--epochs', type=int, default=3); ap.add_argument('--batch-size', type=int, default=32); ap.add_argument('--min-sensitivity', type=float, default=.90); ap.add_argument('--seed', type=int, default=42); args=ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root=Path(args.data_dir); samples, classes=discover(root); train, val, test=stratified_split(samples,args.seed)
    train_loader=DataLoader(RetinaDataset(root,train,True),batch_size=args.batch_size,shuffle=True,num_workers=0)
    val_loader=DataLoader(RetinaDataset(root,val),batch_size=args.batch_size,shuffle=False,num_workers=0)
    test_loader=DataLoader(RetinaDataset(root,test),batch_size=args.batch_size,shuffle=False,num_workers=0)
    num_classes=len(classes)
    light=build_lightweight(num_classes).to(device); expert=build_expert(num_classes).to(device)
    for name, model in [('lightweight',light),('expert',expert)]:
        optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
        for _ in range(args.epochs): train_epoch(model,train_loader,optimizer,device)
        torch.save(model.state_dict(), Path(__file__).resolve().parents[1]/'models'/f'{name}.pt')
    lv, yv, _, light_val_time=predict(light,val_loader,device); ev, _, _, expert_val_time=predict(expert,val_loader,device)
    router=AdaptiveRouter(); router.fit(lv,lv.argmax(1),ev.argmax(1),yv,min_sensitivity=args.min_sensitivity)
    lt, yt, paths, light_test_time=predict(light,test_loader,device); et, _, _, expert_test_time=predict(expert,test_loader,device)
    decision=router.decide(lt); final=np.where(decision.escalate,et.argmax(1),lt.argmax(1))
    expert_only=et.argmax(1); light_only=lt.argmax(1)
    def metrics(pred): return {'accuracy':float(accuracy_score(yt,pred)),'balanced_accuracy':float(balanced_accuracy_score(yt,pred)),'macro_f1':float(f1_score(yt,pred,average='macro',zero_division=0)),'macro_sensitivity':float(recall_score(yt,pred,average='macro',zero_division=0))}
    # Runtime estimate: measured lightweight time for all cases plus measured expert time only for escalated cases.
    estimated=light_test_time + expert_test_time * float(decision.escalate.mean())
    out=Path(__file__).resolve().parents[1]/'results'; out.mkdir(exist_ok=True); (Path(__file__).resolve().parents[1]/'models').mkdir(exist_ok=True)
    result={'classes':classes,'device':str(device),'n_train':len(train),'n_val':len(val),'n_test':len(test),'router_threshold':float(router.threshold),'expert_escalation_rate':float(decision.escalate.mean()),'estimated_adaptive_time_seconds':float(estimated),'measured_lightweight_time_seconds':float(light_test_time),'measured_expert_time_seconds':float(expert_test_time),'lightweight_only':metrics(light_only),'expert_only':metrics(expert_only),'adaptive':metrics(final),'confusion_matrix_adaptive':confusion_matrix(yt,final).tolist()}
    (out/'adaptive_results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    np.save(out/'router_scores.npy',decision.score)
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
