import os
import torch
import unittest
from unittest.mock import patch, MagicMock
from src.train import train

class TestTrain(unittest.TestCase):

    @patch('src.train.os.environ', {'DATA_DIR': 'mock_data_dir', 'OUTPUT_DIR': 'mock_output_dir'})
    @patch('src.train.torch.utils.data.DataLoader')
    def test_train_functionality(self, MockDataLoader):
        # Mock the data loaders to return predefined datasets
        mock_train_loader = MagicMock()
        mock_val_loader = MagicMock()
        MockDataLoader.side_effect = [mock_train_loader, mock_val_loader]

        # Mock the datasets
        mock_train_dataset = MagicMock()
        mock_val_dataset = MagicMock()

        # Mock the data loader return values
        mock_train_loader.__iter__.return_value = iter([('mock_image', 'mock_target')])
        mock_val_loader.__iter__.return_value = iter([('mock_image', 'mock_target')])

        # Call the train function
        train()

        # Check if the data loaders were called with the correct datasets
        MockDataLoader.assert_any_call(mock_train_dataset, batch_size=4, shuffle=True)
        MockDataLoader.assert_any_call(mock_val_dataset, batch_size=4)

    @patch('src.train.os.environ', {'DATA_DIR': 'mock_data_dir', 'OUTPUT_DIR': 'mock_output_dir'})
    @patch('src.train.torch.utils.data.DataLoader')
    def test_empty_dataset(self, MockDataLoader):
        # Mock the data loaders to return empty datasets
        mock_train_loader = MagicMock()
        mock_val_loader = MagicMock()
        MockDataLoader.side_effect = [mock_train_loader, mock_val_loader]

        # Mock the datasets
        mock_train_dataset = MagicMock()
        mock_val_dataset = MagicMock()

        # Mock the data loader return values
        mock_train_loader.__iter__.return_value = iter([])
        mock_val_loader.__iter__.return_value = iter([])

        # Call the train function
        train()

        # Check if the data loaders were called with the correct datasets
        MockDataLoader.assert_any_call(mock_train_dataset, batch_size=4, shuffle=True)
        MockDataLoader.assert_any_call(mock_val_dataset, batch_size=4)

    @patch('src.train.os.environ', {'DATA_DIR': 'mock_data_dir', 'OUTPUT_DIR': 'mock_output_dir'})
    @patch('src.train.torch.utils.data.DataLoader')
    def test_single_class_dataset(self, MockDataLoader):
        # Mock the data loaders to return datasets with only one class
        mock_train_loader = MagicMock()
        mock_val_loader = MagicMock()
        MockDataLoader.side_effect = [mock_train_loader, mock_val_loader]

        # Mock the datasets
        mock_train_dataset = MagicMock()
        mock_val_dataset = MagicMock()

        # Mock the data loader return values
        mock_train_loader.__iter__.return_value = iter([('mock_image', {'labels': [1]})])
        mock_val_loader.__iter__.return_value = iter([('mock_image', {'labels': [1]})])

        # Call the train function
        train()

        # Check if the data loaders were called with the correct datasets
        MockDataLoader.assert_any_call(mock_train_dataset, batch_size=4, shuffle=True)
        MockDataLoader.assert_any_call(mock_val_dataset, batch_size=4)

    @patch('src.train.os.environ', {'DATA_DIR': 'mock_data_dir', 'OUTPUT_DIR': 'mock_output_dir'})
    @patch('src.train.torch.utils.data.DataLoader')
    def test_varying_image_sizes(self, MockDataLoader):
        # Mock the data loaders to return datasets with varying image sizes
        mock_train_loader = MagicMock()
        mock_val_loader = MagicMock()
        MockDataLoader.side_effect = [mock_train_loader, mock_val_loader]

        # Mock the datasets
        mock_train_dataset = MagicMock()
        mock_val_dataset = MagicMock()

        # Mock the data loader return values
        mock_train_loader.__iter__.return_value = iter([('mock_image_1', 'mock_target_1'), ('mock_image_2', 'mock_target_2')])
        mock_val_loader.__iter__.return_value = iter([('mock_image_1', 'mock_target_1'), ('mock_image_2', 'mock_target_2')])

        # Call the train function
        train()

        # Check if the data loaders were called with the correct datasets
        MockDataLoader.assert_any_call(mock_train_dataset, batch_size=4, shuffle=True)
        MockDataLoader.assert_any_call(mock_val_dataset, batch_size=4)

if __name__ == '__main__':
    unittest.main()
