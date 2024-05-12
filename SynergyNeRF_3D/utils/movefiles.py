import os
import shutil
import re

if __name__ == "__main__":
    pattern = r'\d+'
    
    scenes = [
                "chair", 
              "drums", 
              "ficus", 
              "hotdog", 
              "lego", 
              "materials", 
              "mic", 
              "ship"
              ]
    dates=[
            "20231029-053536", 
           "20231029-012703",
           "20231029-060329", 
           "20231029-071124",
           "20231029-063725", 
           "20231029-080525",
           "20231029-074106", 
           "20231029-084140"
           ]
    
    for scene, date in zip(scenes, dates):
        source_dir = f"results/TensoRF_LM_231028_repro/TensoRF_LM_001/TensorVMSplit_{scene}/{date}/render/train_all/white_bg"
        destination_dir = f"/home/mgyukim/datasets/nerf_synthetic_neus_2/{scene}/train/synthetic_train_images_3"
        os.makedirs(destination_dir, exist_ok=True)
        
        img_list = os.listdir(source_dir)
        
        for img in img_list:
            if "gt" in img:
                continue
            
            number = int(re.search(pattern, img).group())
            source_path = os.path.join(source_dir, img)
            
            target_img_name = f"image_{number:03d}.png"
            destination_path = os.path.join(destination_dir, target_img_name)
            
            shutil.copy(source_path, destination_path)