import os
import subprocess
from torch.utils.tensorboard import SummaryWriter

class RunInpaintingWithFID:
    def __init__(self, device, mask_func):
        self.device = device
        self.mask_func = mask_func

    def run(self):
        
        scores = []
        for step in range(1, 30 + 1):
            # 调用 inpainting.py 脚本执行生成任务
            subprocess.run(
                ["python", "inpainting.py", "--mask-func", f"{self.mask_func}"],
                check=True
            )

            # 计算 FID 分数
            fid_score = self.calculate_fid()

            print(f"Step: {step}, FID Score: {fid_score}")
            scores.append(fid_score)
            
        # calculate the avg score and 標準差
        avg_score = sum(scores) / len(scores)
        std_score = (sum([(score - avg_score) ** 2 for score in scores]) / len(scores)) ** 0.5
        return avg_score, std_score

    def calculate_fid(self):
        # cd faster-pytorch-fid/
        # python fid_score_gpu.py --predicted-path ../test_results --device cuda:0
        command = f"cd faster-pytorch-fid && python fid_score_gpu.py --predicted-path ../test_results --device {self.device}"
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            shell=True
        )
        res = result.stdout.strip()
        fid_score = res.split()[-1]
        return float(fid_score)

if __name__ == "__main__":
    
    mode = ["cosine", "linear", "square"]
    
    results = {}
    
    for mask_func in mode:
        print(f"Running mask_func: {mask_func}")
        finder = RunInpaintingWithFID(
            device="cuda:1",
            mask_func=mask_func
        )
        
        avg_score, std_score = finder.run()
        results[mask_func] = (avg_score, std_score)
        
    print(results)
    
    # draw table
    import pandas as pd
    df = pd.DataFrame(results).T
    df.columns = ["avg_score", "std_score"]
    print(df)
    df.to_csv("results.csv")
    df.to_html("results.html")
    
    

    
