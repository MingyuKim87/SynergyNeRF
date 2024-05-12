<div align="center">

# Synergistic Integration of Coordinate Network and Tensorial Feature for Improving NeRFs from Sparse Inputs (ICML2024)

---

</div>

## For dynamic NeRFs
Create a directory named "data" and place the D-NeRF synthetic dataset into this directory.

## Training
```
python main.py config=config/SynergyNeRF/official/{scene}.yaml
```


## Evaluation
When using `render_test=True` and `render_path=True`, the results at test viewpoints are automatically evaluated, and validation viewpoints are generated after the reconstruction process.
Otherwise, if you want to run the code in inference mode, you should execute it as follows. 
```
python main.py python main.py config=config/SynergyNeRF/official/{scene}.yaml systems.ckpt={checkpoint/path} render_only=True
```

## Visualization of Tensorial Features and Coordinate Networks
This implementation offers novel-view synthesis from the learned model and includes visualization of tensorial features as well as the use of only the coordinate-network. Please execute the code as follows. 

```
python main.py python main.py config=config/SynergyNeRF_disentangled/official/{scene}.yaml systems.ckpt={checkpoint/path}
```