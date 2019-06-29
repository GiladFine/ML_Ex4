from gcommand_loader import GCommandLoader
import torch.utils.data as tud

dataset = GCommandLoader('data/test')

test_loader = tud.DataLoader(
        dataset, batch_size=100, shuffle=False,
        num_workers=20, pin_memory=True)

for k, (input,label) in enumerate(test_loader):
    print(input.size(), len(label))
