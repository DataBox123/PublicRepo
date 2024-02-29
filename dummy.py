# import timm
import torch
import math
import time
from collections import deque

class HyperParameters():
    def __init__(self, batch_size=2):
        print("\n---[ HyperParameters ]---")
        self.B = batch_size
        self.C = 3
        self.H = 224
        self.W = 224
        
        self.pred_class = 1000
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.lr = 1e-7
        self.wd = 1e-10
        
        print("batch size:", self.B)
        print("input shape:", self.C, self.H, self.W)
        print("device:", self.device)


class LoopCounter():
    def __init__(self, max=100):
        self.box = {0:0}
        # self.box = {0:0, 1:0}
        self.max = max
        
        
        self.digits = math.ceil(math.log10(self.max))
        if self.digits < 1:
            self.digits = 1
        
        self.effect = deque(["-", "-", "-", " ", " ", " ", " ", " ", " ", " ", " ", " "])
        
    def count(self):
        self.box[0] += 1
        
        # item check
        for i_key in sorted(self.box):
            # print("i_key", i_key)
            if self.box[i_key] >= self.max:
                try:
                    self.box[i_key + 1] += 1
                except:
                    self.box[i_key + 1] = 1
                
                self.box[i_key] = 0
            
    def show(self, line_break=True):
        # 버려짐
        _str = "Count"
        for i_key in sorted(self.box):
            _str += " : " + str(self.box[i_key])
        if line_break:
            print(_str)
        else:
            print("\r"+_str, end="")
    
    def to_str(self, reverse=True, head="Count", split=" : ", effect=True):
        if effect:
            _str = "[" + str("".join(self.effect)) + "] " + head
            self.effect.rotate(1)
        else:
            _str = head
        if reverse:
            list_keys = sorted(self.box)[::-1]
        else:
            list_keys = sorted(self.box)
        
        for i_key in list_keys:
            _str += split + str(self.box[i_key]).rjust(self.digits, " ")
        return _str

class AutomaticMixedPrecisionTool():
    def __init__(self, use_amp=True, detect_anomaly=False):
        print("\n---[ AutomaticMixedPrecisionTool ]---")
        self.use_amp = use_amp
        if self.use_amp:
            self.amp_scaler = torch.cuda.amp.GradScaler(enabled = True)
        else:
            self.amp_scaler = None
        
        self.detect_anomaly = detect_anomaly
        
        print("use_amp:", self.use_amp)
        print("detect_anomaly:", self.detect_anomaly)
        
        
    
    def go_forward(self, model, in_ts, ts_ans, criterion):
        if self.use_amp:
            with torch.cuda.amp.autocast(enabled=True):
                ts_pred = model(in_ts)
                loss = criterion(ts_pred, ts_ans)
                
        else:
            ts_pred = model(in_ts)
            loss = criterion(ts_pred, ts_ans)
        return ts_pred, loss
    
    def go_backward(self, loss, optimizer):
        optimizer.zero_grad()
        if self.use_amp:
            with torch.cuda.amp.autocast(enabled=True):
                if self.detect_anomaly:
                    with torch.autograd.detect_anomaly():
                        self.amp_scaler.scale(loss).backward(retain_graph=False)
                        self.amp_scaler.step(optimizer)
                        self.amp_scaler.update()
                else:
                    self.amp_scaler.scale(loss).backward(retain_graph=False)
                    self.amp_scaler.step(optimizer)
                    self.amp_scaler.update()
        else:
            if self.detect_anomaly:
                with torch.autograd.detect_anomaly():
                    loss.backward()
                    optimizer.step()
            else:
                loss.backward()
                optimizer.step()
        


def train(HP, LC_train, AMPT):
    print("\n---[ init train ]---")
    # HP = HyperParameters()
    # LC_train = LoopCounter()
    # AMPT = AutomaticMixedPrecisionTool(use_amp=True, detect_anomaly=False)
    # AMPT = AutomaticMixedPrecisionTool(use_amp=False, detect_anomaly=False)
    
    try:
        import timm
        list_models = timm.list_models("*davit*")
        # print("list_models", list_models)
        model = timm.create_model(list_models[0])
        print("model: timm", list_models[0])
    except:
        import torchvision.models as models
        model = models.densenet161()
        print("model: torchvision densenet161")
    model.to(HP.device)
    # print("model", model)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=HP.lr, weight_decay=HP.wd)
    model.train()
    
    
    while True:
        LC_train.count()
        time_ = time.time()
        in_ts = torch.randn(HP.B, HP.C, HP.H, HP.W, device=HP.device)
        ts_ans = torch.randn(HP.B, HP.pred_class, device= HP.device)
        
        # print("ts_ans", ts_ans.shape, ts_ans)
        
        ts_pred, loss = AMPT.go_forward(model, in_ts, ts_ans, criterion)
        AMPT.go_backward(loss, optimizer)
        
        # print("loss", loss)
        _str  = LC_train.to_str() 
        _str += " / Loss : " + str(round(loss.item(), 4)).ljust(9, " ")
        _str += " / Time : " + str(round(time.time() - time_, 4)).ljust(6, " ")
        print("\r" + _str, end="")


if __name__ == "__main__":
    import argparse
    parser_options = argparse.ArgumentParser(description='_options')
    parser_options.add_argument("--batch_size", type = int, default = 2,     help = "batch_size")
    args_options = parser_options.parse_args()
    
    LC_main = LoopCounter()
    flag_exit_init = -9
    flag_exit = flag_exit_init
    
    HP = HyperParameters(args_options.batch_size)
    AMPT = AutomaticMixedPrecisionTool()
    # train(HP = HyperParameters()
         # ,LC_train = LoopCounter()
         # ,AMPT = AutomaticMixedPrecisionTool()
         # )
    
    
    
    while True:
        LC_main.count()
        print("\n" + LC_main.to_str(head="Main Loop Count",effect=False)+"\n")
        
        try:
            time.sleep(1)
            flag_exit = -9
            train(HP = HP
                 ,LC_train = LoopCounter()
                 ,AMPT = AMPT
                 )
            
        except:
            flag_exit += 1
            pass
        
        if flag_exit != flag_exit_init:
            if flag_exit >= 0:
                break
            else:
                print(" Cancel directly", - flag_exit , "more times to quit!")
        
    print("\nEOF")
