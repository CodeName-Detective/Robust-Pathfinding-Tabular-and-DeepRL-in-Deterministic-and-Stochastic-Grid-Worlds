import torch

if torch.backends.mps.is_available():
    DEVICE = torch.device(device="mps")
elif torch.cuda.is_available():
    DEVICE = torch.device(device="cuda")
else:
    DEVICE = torch.device(device="cpu")

class ValueFunction(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(ValueFunction, self).__init__()
        self.linear1 = torch.nn.Linear(in_features=input_size, out_features=128)
        self.linear2 = torch.nn.Linear(in_features=128, out_features=output_size)
    
    def forward(self, x):
        x = torch.nn.functional.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    
    def act(self, x):
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.inference_mode():
            return self.forward(x)