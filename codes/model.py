import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, action_size, device):
        """ Create Q-network
        Parameters
        ----------
        action_size: int
            number of actions
        device: torch.device
            device on which to the model will be allocated
        """
        super().__init__()

        self.device = device 
        self.action_size = action_size

        # TODO: Create network
        self.pool = nn.MaxPool2d(2,2)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=9, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=9, out_channels=12, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=12, out_channels=15, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(15*6*6+7, 128)
        self.fc2 = nn.Linear(128, 96)
        self.fc3 = nn.Linear(96, 48)
        self.fc4 = nn.Linear(48, action_size)

    def forward(self, observation):
        """ Forward pass to compute Q-values
        Parameters
        ----------
        observation: np.array
            array of state(s)
        Returns
        ----------
        torch.Tensor
            Q-values  
        """

        # TODO: Forward pass through the network
        observation = torch.tensor(observation).to(self.device)
        speed, abs_sensors, steering, gyroscope = self.extract_sensor_values(observation, observation.shape[0])
        x = observation.reshape((-1,3,96,96))
        x = self.pool(F.leaky_relu(self.conv1(x))) #48
        x = self.pool(F.leaky_relu(self.conv2(x))) #24
        x = self.pool(F.leaky_relu(self.conv3(x))) #12
        x = self.pool(F.leaky_relu(self.conv4(x))) #6
        x = x.view(-1, 15*6*6)
        x = torch.cat([x, speed, abs_sensors, steering, gyroscope], dim=1)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def extract_sensor_values(self, observation, batch_size):
        """ Extract numeric sensor values from state pixels
        Parameters
        ----------
        observation: list
            python list of batch_size many torch.Tensors of size (96, 96, 3)
        batch_size: int
            size of the batch
        Returns
        ----------
        torch.Tensors of size (batch_size, 1),
        torch.Tensors of size (batch_size, 4),
        torch.Tensors of size (batch_size, 1),
        torch.Tensors of size (batch_size, 1)
            Extracted numerical values
        """
        speed_crop = observation[:, 84:94, 13, 0].reshape(batch_size, -1)
        speed = (speed_crop== 255).sum(dim=1, keepdim=True) 
        abs_crop = observation[:, 84:94, 18:25:2, 2].reshape(batch_size, 10, 4)
        abs_sensors = (abs_crop== 255).sum(dim=1) 
        steer_crop = observation[:, 88, 38:58, 1].reshape(batch_size, -1)
        steering = (steer_crop== 255).sum(dim=1, keepdim=True)
        gyro_crop = observation[:, 88, 58:86, 0].reshape(batch_size, -1)
        gyroscope = (gyro_crop== 255).sum(dim=1, keepdim=True)
        
        return speed, abs_sensors.reshape(batch_size, 4), steering, gyroscope