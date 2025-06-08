# TQC + HER for In-Hand Manipulation

```
conda create -n ShadowHand python=3.10
conda activate ShadowHand
pip install -r requirements.txt
```

Train `HandManipulateBlockRotateXYZ-v1` with 16 environments
```
python ShadowHand_TQC.py --env-id HandManipulateBlockRotateXYZ-v1 --seed 4 --num-envs 16
```

Train `HandManipulateBlock_ContinuousTouchSensors-v1` with 16 environments
```
python ShadowHand_TQC.py --env-id HandManipulateBlock_ContinuousTouchSensors-v1 --seed 4 --num-envs 16
```

Each script takes around 8 hours to complete running 10M steps on Hardware of NVIDIA RTX A6000 with 48GB of graphics ram and 16 core CPU.
