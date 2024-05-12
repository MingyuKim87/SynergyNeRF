import os
from PIL import Image

def make_background_transparent(image_path, output_path):
    img = Image.open(image_path)
    img = img.convert("RGBA")
    
    datas = img.getdata()
    
    newData = []
    # for item in datas:
    #     # 흰색 배경이라고 가정할 때, RGB 값이 (255, 255, 255)인 픽셀을 찾아 투명하게 만듭니다.
    #     if item[0] >= 253 and item[1] >= 253 and item[2] >= 253:
    #         # newData.append((255, 255, 255, 0)) # 투명한 픽셀로 변경
    #         newData.append((0, 0, 0, 255)) # 투명한 픽셀로 변경
    #     else:
    #         newData.append(item)
    
    img.putdata(newData)
    img.save(output_path, "PNG")

if __name__ == "__main__":
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
        image_path = f"results/TensoRF_LM_231028_repro/TensoRF_LM_001/TensorVMSplit_{scene}/{date}/render/train_all/"
        new_path = f"results/TensoRF_LM_231028_repro/TensoRF_LM_001/TensorVMSplit_{scene}/{date}/render/train_all/white_bg"
        img_list = os.listdir(image_path)
        img_list = [img for img in img_list if not os.path.isdir(os.path.join(image_path, img))]
        img_list = [img for img in img_list if os.path.splitext(img)[-1] == ".png"]
        os.makedirs(new_path, exist_ok=True)
        
        for img in img_list:
            if "gt" in img:
                continue
            
            img_path = os.path.join(image_path, img)
            output_path = os.path.join(new_path, img)
            
            make_background_transparent(img_path, output_path)
        
    
    