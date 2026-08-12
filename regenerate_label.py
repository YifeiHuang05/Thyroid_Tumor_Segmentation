from pathlib import Path
import argparse,csv,json,shutil
from collections import Counter

PTC_CLASS_ID=1
NON_PTC_CLASS_ID=0
PTC_TERMS={"papillary thyroid carcinoma","micro-papillary thyroid carcinoma","papillary thyroid carcinoma, chronic thyroiditis"}

def norm_pid(x):
    try:return str(int(x)).zfill(8)
    except:return str(x)

def load_json(p):
    with open(p,"r",encoding="utf-8") as f:return json.load(f)

def get_ptc_status(record):
    nodule=record.get("nodule_1") or {}
    hist=str(nodule.get("Histopathology") or "").strip().lower()
    return int(any(term in hist for term in PTC_TERMS))

def build_patient_status(data):
    return {norm_pid(pid):get_ptc_status(record) for pid,record in data["info"].items()}

def load_existing_split(csv_path):
    train,val=set(),set()
    with open(csv_path,"r",encoding="utf-8",newline="") as f:
        for row in csv.DictReader(f):
            pid=norm_pid(row["patient_id"]); split=row["split"].strip().lower()
            if split=="train":train.add(pid)
            elif split=="val":val.add(pid)
    if train & val:raise RuntimeError(f"Patient leakage: {train & val}")
    return train,val

def polygon_to_yolo(seg,w,h,class_id):
    lines=[]
    if not isinstance(seg,list):return lines
    for poly in seg:
        if not isinstance(poly,list) or len(poly)<6 or len(poly)%2:continue
        pts=[]
        for i in range(0,len(poly),2):
            x=max(0,min(1,float(poly[i])/w)); y=max(0,min(1,float(poly[i+1])/h)); pts.extend([x,y])
        lines.append(str(class_id)+" "+" ".join(f"{v:.6f}" for v in pts))
    return lines

def process_split(data,patient_ids,src_img_dir,out_img_dir,out_label_dir,patient_status,split_name):
    patient_ids={norm_pid(x) for x in patient_ids}; out_img_dir.mkdir(parents=True,exist_ok=True); out_label_dir.mkdir(parents=True,exist_ok=True)
    images={x["id"]:x for x in data["images"]}; ann_by_img={}
    for ann in data["annotations"]:ann_by_img.setdefault(ann["image_id"],[]).append(ann)
    stats=Counter()
    for image in data["images"]:
        pid=norm_pid(image["patient_id"])
        if pid not in patient_ids:continue
        fname=image["file_name"]; src=src_img_dir/fname; dst=out_img_dir/fname
        if not src.exists():stats["missing_images"]+=1;continue
        shutil.copy2(src,dst); stats["images"]+=1
        w,h=image["width"],image["height"]; cls=PTC_CLASS_ID if patient_status[pid] else NON_PTC_CLASS_ID
        lines=[]
        for ann in ann_by_img.get(image["id"],[]):
            lines.extend(polygon_to_yolo(ann.get("segmentation"),w,h,cls))
        label_path=out_label_dir/(Path(fname).stem+".txt")
        label_path.write_text("\n".join(lines)+"\n" if lines else "",encoding="utf-8")
        stats["polygons"]+=len(lines); stats["ptc_polygons"]+=sum(line.startswith(f"{PTC_CLASS_ID} ") for line in lines); stats["non_ptc_polygons"]+=sum(line.startswith(f"{NON_PTC_CLASS_ID} ") for line in lines)
    print(f"{split_name}: images={stats['images']} polygons={stats['polygons']} PTC={stats['ptc_polygons']} non-PTC={stats['non_ptc_polygons']} missing={stats['missing_images']}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--train-json", default=None)
    p.add_argument("--train-images", default=None)
    p.add_argument("--test-json", default=None)
    p.add_argument("--test-images", default=None)
    p.add_argument("--split-csv", default=None)
    args = p.parse_args()
    
    source = Path(args.source).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    train_json = Path(args.train_json).expanduser().resolve() if args.train_json else source / "train" / "train_annotations.json"
    train_images = Path(args.train_images).expanduser().resolve() if args.train_images else source / "train" / "images"
    test_json = Path(args.test_json).expanduser().resolve() if args.test_json else source / "test" / "test_annotations.json"
    test_images = Path(args.test_images).expanduser().resolve() if args.test_images else source / "test" / "images"
    split_csv = Path(args.split_csv).expanduser().resolve() if args.split_csv else output / "metadata" / "patient_split.csv"
    
    output.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)
    
    train_data = load_json(train_json)
    test_data = load_json(test_json)
    patient_status = build_patient_status(train_data)
    test_status = build_patient_status(test_data)
    train_patients, val_patients = load_existing_split(split_csv)
    all_train = set(patient_status)
    
    if train_patients | val_patients != all_train:
        raise RuntimeError(
            f"Existing split does not match JSON patients. "
            f"Missing={all_train - (train_patients | val_patients)}, "
            f"extra={(train_patients | val_patients) - all_train}"
        )
    
    process_split(train_data, train_patients, train_images, output / "images" / "train", output / "labels" / "train", patient_status, "TRAIN")
    process_split(train_data, val_patients, train_images, output / "images" / "val", output / "labels" / "val", patient_status, "VAL")
    process_split(test_data, set(test_status), test_images, output / "images" / "test", output / "labels" / "test", test_status, "TEST")
    
    yaml_path = output / "dataset.yaml"
    yaml_content = (
        f"path: {output.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"names:\n"
        f"  0: non-PTC\n"
        f"  1: PTC\n"
    )
    yaml_path.write_text(yaml_content, encoding="utf-8")
    
    print(f"\nDataset YAML: {yaml_path}")
    print(f"PTC patients: {sum(patient_status.values())}")
    print(f"non-PTC patients: {len(patient_status)-sum(patient_status.values())}")
    print("\nDone. Existing patient split was reused; no new split was generated.")

if __name__=="__main__":main()