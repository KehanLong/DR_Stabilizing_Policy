import torch
import torch.nn as nn
import torch.optim as optim

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()


        self.fc1 = nn.Linear(3, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.origin_tensor = torch.zeros(1, 3).to(self.fc1.weight.device)

    def forward(self, input):



        x = self.fc1(input)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        phi_x = self.fc3(x)

        #print(f"phi_x shape: {phi_x.shape}")
        phi_0 = self.fc3(self.relu2(self.fc2(self.relu1(self.fc1(self.origin_tensor)))))
        #print(f"phi_0 shape: {phi_0.shape}")

        V = phi_x 

        return V

def negative_definite_func(x):
    return torch.pow(x, 2) + 0.01

def train_network(model, num_epochs=10000, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)


    for epoch in range(num_epochs):
        x = torch.randn(500, 3).to(model.fc1.weight.device)
        y_target = negative_definite_func(x)

        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y_target)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    return model

def create_simple_model():
    model = SimpleModel()

    model.load_state_dict(torch.load("trained_model.pth"))
    return model

def box_data(lower_limit, upper_limit, scale):
    data_min = scale * torch.tensor(lower_limit, dtype=torch.get_default_dtype()).unsqueeze(0)
    data_max = scale * torch.tensor(upper_limit, dtype=torch.get_default_dtype()).unsqueeze(0)
    # device = torch.device("cuda")
    # data_min = data_min.to(device)
    # data_max = data_max.to(device)
    return {
        'data_min': data_min,
        'data_max': data_max,
    }


def main():
    model = SimpleModel()
    # device = torch.device("cuda")
    # model.to(device)
    trained_model = train_network(model)

    torch.save(trained_model.state_dict(), "trained_model.pth")

if __name__ == "__main__":
    main()
