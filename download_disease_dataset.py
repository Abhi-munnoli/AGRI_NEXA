from datasets import load_dataset
from pathlib import Path
from PIL import Image
import re

SELECTED=["Tomato___Early_blight","Tomato___Late_blight","Tomato___healthy",
"Potato___Early_blight","Potato___Late_blight","Potato___healthy",
"Corn___Common_rust","Corn___Northern_Leaf_Blight","Corn___healthy"]

out=Path("disease_dataset"); out.mkdir(exist_ok=True)
ds=load_dataset("mohanty/PlantVillage","color",split="train")
labels=ds.features["label"].names
wanted=set(SELECTED)
found=set(labels)&wanted
missing=wanted-found
if missing: raise RuntimeError("Missing classes: "+", ".join(sorted(missing)))

counts={x:0 for x in SELECTED}
for i,row in enumerate(ds):
    label=labels[row["label"]]
    if label not in wanted: continue
    folder=out/re.sub(r"[^A-Za-z0-9]+","_",label).strip("_")
    folder.mkdir(parents=True,exist_ok=True)
    row["image"].convert("RGB").save(folder/f"{i:06d}.jpg",quality=95)
    counts[label]+=1
print("Downloaded selected classes:")
for k,v in counts.items(): print(k,v)
