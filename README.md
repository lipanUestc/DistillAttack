## 1、Train Watermarked Models

```bash
cd Watermark/
python Entangled/train.py --dataset cifar10 --default 1
python MEA-Defender/secure_train.py --composite_class_A=0 --composite_class_B=1 --target_class=2 --epoch=100
...
```

## 2、Execute GuidExt Attack

```bash
python model_distillation.py 
```

## 3、Model Modification Attack

```bash
python finetune.py
python prune.py
python potential_attack.py
...
```

## 4、Draw Distribution Plot

```bash
# generation
python draw_dist.py
python fluctuate.py
```

