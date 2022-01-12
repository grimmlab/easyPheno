import torch


class BaseModel:
    """
    Class for BaseModel parent class for all models that can be used within the framework
    """

    def __init__(self, model_name: str, task: str):
        """
        Init for Base Model
        :param model_name: name of the model
        :param task: task (regression or classification)
        """
        self.name = model_name
        self.task = task  #auto identification einbauen welcher Task gefordert ist anhand der Zielvariable
        if self.task == 'classification':
            self.loss = torch.nn.CrossEntropyLoss
        else:
            self.loss = torch.nn.MSELoss  #tbd. do we want to fix the loss function to use or also adjust it?
        self.model = self.define_model()
        self.optimizer = self.define_optimizer()

    def define_model(self) -> torch.nn.Sequential:
        """
        Method for defining the model (architecture and parameter ranges). Needs to be implemented by every child class.
        :return: Sequential model
        """
        raise NotImplementedError

    def define_optimizer(self) -> torch.optim.optimizer:
        """
        Method for defining the optimizer for the model.
        :return: optimizer
        """
        raise NotImplementedError

    def train_one_epoch(self, train_data_loader: torch.utils.data.DataLoader, device: torch.device):
        self.model.train()
        for batch_idx, (inputs, targets) in enumerate(train_data_loader):
            inputs, targets = inputs.view(inputs.size(0), -1).to(device), targets.to(device)
            self.optimizer.zero_grad()
            output = self.model(inputs.float())
            loss = self.loss_fn(output, torch.reshape(targets, (-1, 1)).float())
            loss.backward()
            self.optimizer.step()
