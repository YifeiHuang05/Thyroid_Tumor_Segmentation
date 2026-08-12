from pathlib import Path
import argparse,csv,time
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
from sklearn.metrics import accuracy_score,balanced_accuracy_score,confusion_matrix,f1_score,precision_score,recall_score

def parse_args():
    parser=argparse.ArgumentParser(description="YOLO26n-seg training, optional HPO, and PTC classification evaluation for ThyroidXL.")
    parser.add_argument("--dataset",required=True,help="Path to dataset.yaml")
    parser.add_argument("--model",required=True,help="Path to local yolo26n-seg.pt")
    parser.add_argument("--project",default=None,help="Output project directory")
    parser.add_argument("--epochs",type=int,default=100,help="Final training epochs")
    parser.add_argument("--batch",type=int,default=-1,help="Batch size. -1 lets Ultralytics auto-select.")
    parser.add_argument("--device",default="0",help="GPU device, e.g. 0 or 0,1")
    parser.add_argument("--workers",type=int,default=8,help="DataLoader workers")
    parser.add_argument("--run-tune",action="store_true",help="Run Ultralytics model.tune() before final training.")
    parser.add_argument("--tune-iterations",type=int,default=10,help="Number of HPO iterations.")
    parser.add_argument("--tune-epochs",type=int,default=20,help="Epochs per HPO trial.")
    parser.add_argument("--conf",type=float,default=0.25,help="Confidence threshold for PTC classification.")
    parser.add_argument("--test",action="store_true",help="Evaluate final model on test set.")
    return parser.parse_args()

def check_cuda(device):
    print("="*70);print("SYSTEM CHECK");print("="*70)
    print(f"PyTorch version:       {torch.__version__}")
    print(f"PyTorch CUDA version:  {torch.version.cuda}")
    print(f"CUDA available:        {torch.cuda.is_available()}")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is not available. Do not start training until PyTorch can see the GPU.")
    print(f"GPU:                   {torch.cuda.get_device_name(0)}")
    print(f"Requested device:      {device}")
    print("="*70)

def train_model(model_path,dataset,project,epochs,batch,device,workers,name,hyperparameters=None):
    print("\n"+"="*70);print(f"TRAINING: {name}");print("="*70)
    model=YOLO(str(model_path))
    train_args={"data":str(dataset),"epochs":epochs,"batch":batch,"device":device,"workers":workers,"project":str(project),"name":name,"exist_ok":True,"pretrained":True,"plots":True,"save":True,"val":True}
    if hyperparameters: train_args.update(hyperparameters)
    print("\nTraining arguments:")
    for key,value in train_args.items(): print(f"  {key}: {value}")
    results=model.train(**train_args)
    return model,results

def run_tuning(model_path,dataset,project,iterations,epochs,batch,device,workers):
    print("\n"+"="*70);print("YOLO26n-seg HYPERPARAMETER TUNING");print("="*70)
    print(f"Iterations: {iterations}");print(f"Epochs/trial: {epochs}");print("NOTE: this uses the validation set for HPO.")
    model=YOLO(str(model_path))
    tune_dir=Path(project)/"tune"
    model.tune(data=str(dataset),epochs=epochs,iterations=iterations,batch=batch,device=device,workers=workers,project=str(project),name="tune",plots=True,save=True,val=True)
    best_yaml=tune_dir/"best_hyperparameters.yaml"
    if not best_yaml.exists(): raise FileNotFoundError(f"Could not find:\n{best_yaml}\nCheck the tuning output directory.")
    print(f"\nBest hyperparameters:\n{best_yaml}")
    return best_yaml

def load_yaml(path):
    import yaml
    with open(path,"r",encoding="utf-8") as f: return yaml.safe_load(f)

def standard_validation(model_path,dataset,device,project):
    print("\n"+"="*70);print("ULTRALYTICS STANDARD TEST EVALUATION");print("="*70)
    model=YOLO(str(model_path))
    metrics=model.val(data=str(dataset),split="test",device=device,plots=True,project=str(project),name="test_metrics",exist_ok=True)
    print("\nStandard Ultralytics metrics:")
    try:
        print(f"Box mAP50:        {metrics.box.map50:.4f}")
        print(f"Box mAP50-95:     {metrics.box.map:.4f}")
    except Exception: pass
    try:
        print(f"Mask mAP50:       {metrics.seg.map50:.4f}")
        print(f"Mask mAP50-95:    {metrics.seg.map:.4f}")
    except Exception: pass
    return metrics

def load_ground_truth_labels(dataset_root):
    """Ground truth PTC status from YOLO segmentation labels: 0=non-PTC, 1=PTC."""
    image_dir=Path(dataset_root)/"images"/"test"
    label_dir=Path(dataset_root)/"labels"/"test"
    ground_truth={}
    for image_path in sorted(image_dir.iterdir()):
        if not image_path.is_file(): continue
        label_path=label_dir/f"{image_path.stem}.txt"
        if not label_path.exists(): raise FileNotFoundError(f"Missing label file:\n{label_path}")
        classes=[]
        with open(label_path,"r",encoding="utf-8") as f:
            for line in f:
                line=line.strip()
                if line: classes.append(int(line.split()[0]))
        if not classes: raise RuntimeError(f"No segmentation annotation found for {image_path}. Every nodule should have a class label.")
        if any(c not in (0,1) for c in classes): raise RuntimeError(f"Unexpected class ID in {label_path}: {classes}")
        ground_truth[image_path.stem]=int(1 in classes)
    return ground_truth

def image_level_evaluation(model_path,dataset_root,device,conf,project):
    print("\n"+"="*70);print("IMAGE-LEVEL PTC CLASSIFICATION");print("="*70)
    model=YOLO(str(model_path))
    ground_truth=load_ground_truth_labels(dataset_root)
    image_dir=Path(dataset_root)/"images"/"test"
    y_true=[];y_pred=[];inference_times=[];output_rows=[]
    for image_path in sorted(image_dir.iterdir()):
        if not image_path.is_file(): continue
        stem=image_path.stem
        if torch.cuda.is_available(): torch.cuda.synchronize()
        start=time.perf_counter()
        results=model.predict(source=str(image_path),conf=conf,device=device,verbose=False,save=False)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        inference_ms=(time.perf_counter()-start)*1000
        inference_times.append(inference_ms)
        result=results[0]
        predicted_classes=[]
        if result.boxes is not None and len(result.boxes)>0:
            predicted_classes=[int(c) for c in result.boxes.cls.cpu().numpy()]
        pred=int(1 in predicted_classes)
        gt=ground_truth[stem]
        y_true.append(gt);y_pred.append(pred)
        output_rows.append({"image":image_path.name,"ground_truth":gt,"prediction":pred,"n_predictions":len(predicted_classes),"predicted_classes":",".join(map(str,predicted_classes)),"inference_ms":inference_ms})

    accuracy=accuracy_score(y_true,y_pred)
    balanced_accuracy=balanced_accuracy_score(y_true,y_pred)
    f1=f1_score(y_true,y_pred,zero_division=0)
    macro_f1=f1_score(y_true,y_pred,average="macro",zero_division=0)
    precision=precision_score(y_true,y_pred,zero_division=0)
    recall=recall_score(y_true,y_pred,zero_division=0)
    cm=confusion_matrix(y_true,y_pred,labels=[0,1])
    tn,fp,fn,tp=cm.ravel()
    specificity=tn/(tn+fp) if (tn+fp)>0 else 0.0
    sensitivity=tp/(tp+fn) if (tp+fn)>0 else 0.0
    mean_inference_ms=float(np.mean(inference_times))
    median_inference_ms=float(np.median(inference_times))
    p95_inference_ms=float(np.percentile(inference_times,95))

    print("\nImage-level PTC metrics:")
    print(f"Accuracy:             {accuracy:.4f}")
    print(f"Balanced accuracy:    {balanced_accuracy:.4f}")
    print(f"Macro F1:             {macro_f1:.4f}")
    print(f"F1:                   {f1:.4f}")
    print(f"Precision:            {precision:.4f}")
    print(f"Sensitivity/Recall:   {sensitivity:.4f}")
    print(f"Specificity:          {specificity:.4f}")
    print("\nConfusion matrix:")
    print("                 Predicted")
    print("                 Negative Positive")
    print(f"Actual Negative  {tn:8d} {fp:8d}")
    print(f"Actual Positive  {fn:8d} {tp:8d}")
    print("\nInference time:")
    print(f"Mean:               {mean_inference_ms:.2f} ms/image")
    print(f"Median:             {median_inference_ms:.2f} ms/image")
    print(f"P95:                {p95_inference_ms:.2f} ms/image")

    output_dir=Path(project)/"test_metrics"
    output_dir.mkdir(parents=True,exist_ok=True)
    csv_path=output_dir/"image_level_predictions.csv"
    with open(csv_path,"w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=["image","ground_truth","prediction","n_predictions","predicted_classes","inference_ms"])
        writer.writeheader();writer.writerows(output_rows)

    metrics_path=output_dir/"image_level_metrics.csv"
    with open(metrics_path,"w",newline="",encoding="utf-8") as f:
        writer=csv.writer(f);writer.writerow(["metric","value"])
        for k,v in {"accuracy":accuracy,"balanced_accuracy":balanced_accuracy,"macro_f1":macro_f1,"f1":f1,"precision":precision,"sensitivity":sensitivity,"specificity":specificity,"tn":tn,"fp":fp,"fn":fn,"tp":tp,"mean_inference_ms":mean_inference_ms,"median_inference_ms":median_inference_ms,"p95_inference_ms":p95_inference_ms}.items(): writer.writerow([k,v])

    np.savetxt(output_dir/"confusion_matrix.csv",cm,delimiter=",",fmt="%d")
    plt.figure(figsize=(8,6));sns.heatmap(cm,annot=True,fmt="d",xticklabels=["non-PTC","PTC"],yticklabels=["non-PTC","PTC"]);plt.title("PTC Classification Confusion Matrix");plt.xlabel("Predicted");plt.ylabel("Actual");plt.tight_layout();plt.savefig(output_dir/"confusion_matrix.png",dpi=300);plt.close()
    print("\nSaved:")
    print(f"  {csv_path}")
    print(f"  {metrics_path}")
    print(f"  {output_dir/'confusion_matrix.csv'}")
    print(f"  {output_dir/'confusion_matrix.png'}")
    return {"accuracy":accuracy,"balanced_accuracy":balanced_accuracy,"macro_f1":macro_f1,"f1":f1,"precision":precision,"sensitivity":sensitivity,"specificity":specificity,"mean_inference_ms":mean_inference_ms,"median_inference_ms":median_inference_ms,"p95_inference_ms":p95_inference_ms,"confusion_matrix":cm}

def main():
    args=parse_args()
    dataset=Path(args.dataset).resolve()
    model_path=Path(args.model).resolve()
    if not dataset.exists(): raise FileNotFoundError(f"dataset.yaml not found:\n{dataset}")
    if not model_path.exists(): raise FileNotFoundError(f"Model not found:\n{model_path}")
    dataset_root=dataset.parent
    project=Path(args.project).resolve() if args.project else dataset_root/"runs"
    project.mkdir(parents=True,exist_ok=True)
    check_cuda(args.device)

    best_hyperparameters=None
    if args.run_tune:
        best_yaml=run_tuning(model_path,dataset,project,args.tune_iterations,args.tune_epochs,args.batch,args.device,args.workers)
        best_hyperparameters=load_yaml(best_yaml)
        print("\nUsing tuned hyperparameters:")
        for key,value in best_hyperparameters.items(): print(f"  {key}: {value}")

    final_model,_=train_model(model_path,dataset,project,args.epochs,args.batch,args.device,args.workers,"final",best_hyperparameters)
    best_model_path=project/"final"/"weights"/"best.pt"
    print(f"\nBest model:\n{best_model_path}")

    if args.test:
        standard_validation(best_model_path,dataset,args.device,project)
        image_level_evaluation(best_model_path,dataset_root,args.device,args.conf,project)

    print("\n"+"="*70);print("PIPELINE COMPLETE");print("="*70)

if __name__=="__main__":
    main()